use std::{
    borrow::Cow,
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    fs::File,
    io::BufWriter,
    io::Write,
    iter, mem,
    ops::{BitOr, Sub},
};

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use chunky::{ChunkFlags, ChunkId, ChunkyFile};
use embedded_graphics_core::prelude::RgbColor;
use gltf::{
    binary::Header,
    buffer::Target,
    json::{
        accessor::{ComponentType, GenericComponentType, Type},
        buffer::View,
        material::{EmissiveFactor, PbrBaseColorFactor, PbrMetallicRoughness},
        mesh::Primitive,
        scene::UnitQuaternion,
        texture::{Info, Sampler},
        validation::Checked,
        Accessor, Asset, Buffer, Image, Index, Material, Mesh, Node, Root, Scene, Texture,
    },
    material::AlphaMode,
    mesh::Mode,
    texture::{MagFilter, MinFilter, WrappingMode},
    Glb, Semantic,
};
use lazy_static::lazy_static;
use maplit::hashmap;
use memmap2::Mmap;
use nalgebra::{point, Matrix, Matrix4, Point2, Scalar, Scale3, Translation3};
use order::Loader;
use png::{BitDepth, ColorType, Encoder};
use rayon::prelude::*;
use rectangle_pack::{
    GroupedRectsToPlace, PackedLocation, RectToInsert, RectanglePackError, RectanglePackOk,
    TargetBin,
};
use serde_json::{Number, Value};
use tinybmp::RawBmp;

use crate::{
    ggcl::AnimationCells, ggcm::Costumes, glbs::BodyPartSets, glpi::Armature,
    glxf::AnimationTransforms, modl::Model, tmap::TextureMap, tmpl::Template,
    txxf::TextureTransform,
};

mod brender;
mod chunky;
mod ggcl;
mod ggcm;
mod ggf;
mod glbs;
mod glf;
mod glpi;
mod glxf;
mod kauai;
mod mbmp;
mod modl;
mod mtrl;
mod order;
mod tmap;
mod tmpl;
mod txxf;

struct TemplateData {
    armature: Armature,
    body_part_sets: BodyPartSets,
    models: HashMap<u32, ModelData>,
    costumes: Costumes,
    materials: HashMap<u32, CustomMaterialData>,
    action_cells: AnimationCells,
    action_transforms: AnimationTransforms,
}

struct ModelData {
    model: Model,
}

struct CustomMaterialData {
    accessories: HashMap<u32, Model>,
    textures: HashMap<ChunkId, TextureMap>,
    parts: Vec<MaterialData>,
}

struct MaterialData {
    material: mtrl::Material,
    texture_map: Option<ChunkId>,
    texture_transform: Option<txxf::TextureTransform>,
}

fn main() -> Result<()> {
    let tmpls = File::open("../3DMMForever/content-files/tmpls.3cn")?;
    let tmpls = unsafe { Mmap::map(&tmpls)? };
    let tmpls = ChunkyFile::load(&tmpls[..])?;

    tmpls.index.par_iter().filter(|(key, value)| key.tag == "TMPL" && value.flags.contains(ChunkFlags::LONER) /* && value.name == "Willy" */).try_for_each(|(_, value)| {
        dbg!(&value.name);
        let data = tmpls.get_chunk(value)?;
        let _tmpl = Template::load(&data)?;

        let Some(armature) = value.get_child(0, "GLPI").and_then(|c| tmpls.index.get(c)) else {
            bail!("No GLPI");
        };
        let armature = Armature::load(&tmpls.get_chunk(armature)?)?;

        let Some(body_part_sets) = value.get_child(0, "GLBS").and_then(|c| tmpls.index.get(c)) else {
            bail!("No GLBS");
        };
        let body_part_sets = BodyPartSets::load(&tmpls.get_chunk(body_part_sets)?)?;

        let Some(costumes) = value.get_child(0, "GGCM").and_then(|c| tmpls.index.get(c)) else {
            bail!("No GGCM");
        };
        let costumes = tmpls.get_chunk(costumes)?;
        let costumes = Costumes::load(&costumes)?;

        let materials: HashMap<u32, CustomMaterialData> = costumes.part_sets.par_iter().enumerate().flat_map(|(set_index, set)| {
            let group_size = body_part_sets
                .groups
                .iter()
                .filter(|g| **g as usize == set_index)
                .count();

            let tmpls = &tmpls;
            set.par_iter().map(move |material_index| {
                let Some(custom_material) = value.get_child(*material_index, "CMTL").and_then(|c| tmpls.index.get(c)) else {
                    bail!("Missing CMTL {material_index}");
                };
                let mut accessories = HashMap::new();
                for child in custom_material.children.iter().filter(|c| c.chunk_id.tag == "BMDL") {
                    let Some(chunk) = tmpls.index.get(&child.chunk_id) else {
                        bail!("Missing accessory data {child:?} for material {material_index}");
                    };
                    let model = Model::load(&tmpls.get_chunk(chunk)?)?;
                    accessories.insert(child.child_id, model);
                }

                let load_material = |part_index| {
                    let Some(material_chunk) = custom_material.get_child(part_index, "MTRL").and_then(|c| tmpls.index.get(c)) else {
                        bail!("Missing material {material_index} {part_index}");
                    };
                    let material = mtrl::Material::load(&tmpls.get_chunk(material_chunk)?)?;
                    let texture_map = material_chunk.get_child(0, "TMAP").copied();

                    let texture_transform = if let Some(texture_transform) =
                        material_chunk.get_child(0, "TXXF")
                    {
                        let Some(texture_transform) = tmpls.index.get(texture_transform) else {
                            bail!("Missing texture transform for material {material_index} {part_index}");
                        };
                        Some(txxf::TextureTransform::load(
                            &tmpls.get_chunk(texture_transform)?,
                        )?)
                    } else {
                        None
                    };

                    Ok(MaterialData {
                        material,
                        texture_map,
                        texture_transform,
                    })
                };

                let parts = (0..group_size).into_par_iter().map(|part_index| {
                    load_material(part_index as u32).with_context(|| {
                        format!("Loading material {material_index} {part_index}")
                    })
                }).collect::<Result<Vec<_>>>()?;

                let textures = parts.par_iter().filter_map(|p| p.texture_map).map(|texture_id| {
                    let Some(texture_map) = tmpls.index.get(&texture_id) else {
                        bail!("Missing texture map for material {material_index}");
                    };
                    Ok((texture_id, TextureMap::load(&tmpls.get_chunk(texture_map)?)?))
                }).collect::<Result<HashMap<_, _>>>()?;

                Ok((*material_index, CustomMaterialData {
                    accessories,
                    parts,
                    textures,
                }))
            })
        }).collect::<Result<HashMap<_, _>>>()?;

        let models = value
            .children
            .par_iter()
            .filter(|c| c.chunk_id.tag == "BMDL")
            .map(|model_link| {
                let Some(model) = tmpls.index.get(&model_link.chunk_id) else {
                bail!("Missing model {}", model_link.child_id);
            };
                let model = Model::load(&tmpls.get_chunk(model)?)?;
                Ok((model_link.child_id, ModelData { model }))
            })
            .collect::<Result<HashMap<u32, ModelData>>>()?;

        let Some(action) = value.get_child(0, "ACTN").and_then(|c| tmpls.index.get(c)) else {
            bail!("No default action");
        };
        let Some(action_cells) = action.get_child(0, "GGCL").and_then(|c| tmpls.index.get(c)) else {
            bail!("No default action GGCL");
        };
        let action_cells = AnimationCells::load(&tmpls.get_chunk(action_cells)?)?;
        let Some(action_transforms) = action.get_child(0, "GLXF").and_then(|c| tmpls.index.get(c)) else {
            bail!("No default action GLXF");
        };
        let action_transforms =
            AnimationTransforms::load(&tmpls.get_chunk(action_transforms)?)?;

        let mut template = TemplateData {
            armature,
            body_part_sets,
            models,
            costumes,
            materials,
            action_cells,
            action_transforms,
        };

        pack_textures(&value.name, &mut template)?;

        export_model(
            &value.name,
            &template,
        )?;

        Ok(())
    })?;

    Ok(())
}

pub trait GetMut<T> {
    fn get_mut(&mut self, id: Index<T>) -> Option<&mut T>;
}

impl GetMut<Node> for Root {
    fn get_mut(&mut self, id: Index<Node>) -> Option<&mut Node> {
        self.nodes.get_mut(id.value())
    }
}

fn pack_textures(name: &str, template: &mut TemplateData) -> Result<()> {
    for (set, set_costumes) in template.costumes.part_sets.iter().enumerate() {
        let set = set as u16;

        #[derive(Copy, Clone, Debug)]
        struct Bounds<T>
        where
            T: Clone + Scalar,
        {
            min: Point2<T>,
            max: Point2<T>,
        }

        impl<T> Bounds<T>
        where
            T: Copy + Scalar,
        {
            fn width(&self) -> T::Output
            where
                T: Sub,
            {
                self.max.x - self.min.x
            }
            fn height(&self) -> T::Output
            where
                T: Sub,
            {
                self.max.y - self.min.y
            }
        }

        impl BitOr for Bounds<u32> {
            type Output = Bounds<u32>;

            fn bitor(self, rhs: Self) -> Self::Output {
                Bounds {
                    min: point![self.min.x.min(rhs.min.x), self.min.y.min(rhs.min.y)],
                    max: point![self.max.x.max(rhs.max.x), self.max.y.max(rhs.max.y)],
                }
            }
        }

        impl BitOr for Bounds<f64> {
            type Output = Bounds<f64>;

            fn bitor(self, rhs: Self) -> Self::Output {
                Bounds {
                    min: point![self.min.x.min(rhs.min.x), self.min.y.min(rhs.min.y)],
                    max: point![self.max.x.max(rhs.max.x), self.max.y.max(rhs.max.y)],
                }
            }
        }

        fn find_pixel_extents(
            model: &Model,
            txxf: &TextureTransform,
            texture_map: &TextureMap,
        ) -> Bounds<u32> {
            let mut local_vertex_extents = Bounds {
                min: model.vertices[0].map,
                max: model.vertices[0].map,
            };
            for vertex in &model.vertices[1..] {
                local_vertex_extents = local_vertex_extents
                    | Bounds {
                        min: vertex.map,
                        max: vertex.map,
                    };
            }
            let local_vertex_extents = Bounds {
                min: point![local_vertex_extents.min.x, local_vertex_extents.min.y],
                max: point![local_vertex_extents.max.x, local_vertex_extents.max.y],
            };
            let local_pixel_extents = Bounds {
                min: txxf
                    .transform_point(&local_vertex_extents.min)
                    .coords
                    .component_mul(
                        &point![texture_map.width as f64, texture_map.height as f64].coords,
                    )
                    .into(),
                max: txxf
                    .transform_point(&local_vertex_extents.max)
                    .coords
                    .component_mul(
                        &point![texture_map.width as f64, texture_map.height as f64].coords,
                    )
                    .into(),
            };
            Bounds {
                min: point![
                    local_pixel_extents.min.x.floor() as u32,
                    local_pixel_extents.min.y.floor() as u32
                ],
                max: point![
                    local_pixel_extents.max.x.ceil() as u32,
                    local_pixel_extents.max.y.ceil() as u32
                ],
            }
        }

        // Determine the pixel size of every material and the used extent of every texture.
        let mut valid_costumes = vec![set_costumes[0]];
        let first_custom_material = &template.materials[&set_costumes[0]];
        let mut first_custom_sizes = Vec::new();
        let mut texture_order = Vec::new();
        let mut texture_extents = HashMap::new();
        for (part, (_, cps)) in template.action_cells.cells[0]
            .parts
            .iter()
            .enumerate()
            .filter(|(i, _)| template.body_part_sets.groups[*i] == set)
            .enumerate()
        {
            let Some(model_id) = cps.model_id else {
                first_custom_sizes.push(None);
                continue;
            };
            let first_material = &first_custom_material.parts[part];
            let Some(first_texture_map_id) = &first_material.texture_map else {
                first_custom_sizes.push(None);
                continue;
            };
            let model = &template.models[&(model_id as u32)];
            if model.model.vertices.is_empty() {
                first_custom_sizes.push(None);
                continue;
            }
            let first_texture_map = &first_custom_material.textures[first_texture_map_id];
            let txxf = first_material.texture_transform.unwrap_or_default();
            let first_size = Point2::from(
                txxf.transform_point(&point![1.0, 1.0])
                    .coords
                    .component_mul(
                        &point![
                            first_texture_map.width as f64,
                            first_texture_map.height as f64
                        ]
                        .coords,
                    ),
            );
            assert!(first_size.x >= 0.0);
            assert!(first_size.y >= 0.0);
            first_custom_sizes.push(Some(point![
                first_size.x.ceil() as u32,
                first_size.y.ceil() as u32
            ]));

            let local_pixel_extents = find_pixel_extents(&model.model, &txxf, first_texture_map);
            match texture_extents.entry(first_texture_map_id) {
                Entry::Vacant(e) => {
                    e.insert(local_pixel_extents);
                    texture_order.push(first_texture_map_id);
                }
                Entry::Occupied(mut e) => {
                    let v = e.get_mut();
                    *v = *v | local_pixel_extents;
                }
            }
        }

        // Ensure all the materials being put into the atlas have the same input sizes across all costumes.
        'costumes: for &costume in &set_costumes[1..] {
            let custom_material = &template.materials[&costume];
            for (part, (_, cps)) in template.action_cells.cells[0]
                .parts
                .iter()
                .enumerate()
                .filter(|(i, _)| template.body_part_sets.groups[*i] == set)
                .enumerate()
            {
                if cps.model_id.is_none() {
                    continue;
                }
                let Some(first_size) = first_custom_sizes[part] else {
                    continue;
                };
                let material = &custom_material.parts[part];
                let Some(texture_map) = &material.texture_map else {
                    continue;
                };
                let texture_map = &custom_material.textures[texture_map];
                let size = Point2::from(
                    material
                        .texture_transform
                        .unwrap_or_default()
                        .transform_point(&point![1.0, 1.0])
                        .coords
                        .component_mul(
                            &point![texture_map.width as f64, texture_map.height as f64].coords,
                        ),
                );
                let size = point![size.x.ceil() as u32, size.y.ceil() as u32];
                if size != first_size {
                    // This could be fixed by generating a second texcoord stream.
                    // It's not a problem for any of the standard actors.
                    eprintln!("Unsupported in {name}: part {part} has an incompatible texture transform in costume {costume} compared to {} ({size} vs {first_size})", set_costumes[0]);
                    continue 'costumes;
                }
            }
            valid_costumes.push(costume);
        }

        fn pack_to_minimal_square(
            rects_to_place: GroupedRectsToPlace<ChunkId, u32>,
        ) -> Result<(u32, RectanglePackOk<ChunkId, u32>)> {
            const SIZES: &[u32] = &[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
            let mut possible_sizes = SIZES;
            let mut packed = None;
            loop {
                let (left, right) = possible_sizes.split_at(possible_sizes.len() / 2);
                let Some((middle, right)) = right.split_first() else {
                    break;
                };
                let mut target_bins = BTreeMap::new();
                target_bins.insert(0, TargetBin::new(*middle, *middle, 1));
                match rectangle_pack::pack_rects(
                    &rects_to_place,
                    &mut target_bins,
                    &rectangle_pack::volume_heuristic,
                    &rectangle_pack::contains_smallest_box,
                ) {
                    Ok(ok) => {
                        packed = Some((*middle, ok));
                        possible_sizes = left;
                    }
                    Err(RectanglePackError::NotEnoughBinSpace) => {
                        possible_sizes = right;
                    }
                }
            }
            let Some((size, layout)) = packed else {
                bail!("Unable to pack into 4k? {rects_to_place:?}");
            };
            Ok((size, layout))
        }

        fn blit(
            texture_map: &TextureMap,
            extents: &Bounds<u32>,
            canvas: &mut [u8],
            size: u32,
            location: &PackedLocation,
        ) {
            let mut height = 0;
            let source_rows = (extents.min.y..extents.max.y)
                .map(|y| y.clamp(0, texture_map.height as u32 - 1))
                .map(|y| {
                    &texture_map.data[texture_map.width as usize * y as usize
                        ..texture_map.width as usize * (y as usize + 1)]
                });
            let dest_rows = canvas
                .chunks_exact_mut(size as usize)
                .skip(location.y() as usize);
            for (source_row, dest_row) in source_rows.zip(dest_rows) {
                height += 1;
                let mut width = 0;
                for (dest_x, source_x) in (extents.min.x..extents.max.x).enumerate() {
                    width += 1;
                    let source_x = source_x.clamp(0, texture_map.width as u32 - 1);
                    let dest_x = dest_x + location.x() as usize;
                    dest_row[dest_x] = source_row[source_x as usize];
                }
                assert_eq!(width, location.width());
            }
            assert_eq!(height, location.height());
        }

        fn write_png<W>(data: &[u8], width: u32, height: u32, writer: W) -> Result<()>
        where
            W: Write,
        {
            let mut encoder = Encoder::new(writer, width, height);
            encoder.set_color(ColorType::Indexed);
            encoder.set_palette(&*PALETTE);
            encoder.set_depth(BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(data)?;
            writer.finish()?;
            Ok(())
        }

        // If texture_order is empty, that means this custom material is for accessories (or the material does not use textures).
        // Accessory custom materials have unique meshes so each one has its own atlas.
        if texture_order.is_empty() {
            for &costume in set_costumes {
                let mut rects_to_place = GroupedRectsToPlace::<_, u32>::new();
                let custom_material = template.materials.get_mut(&costume).unwrap();
                let mut texture_extents = HashMap::new();
                for (part_id, material) in custom_material.parts.iter().enumerate() {
                    let Some(model) = custom_material.accessories.get(&(part_id as u32)) else {
                        continue;
                    };
                    if model.vertices.is_empty() {
                        continue;
                    }
                    let Some(texture_map_id) = &material.texture_map else {
                        continue;
                    };
                    let texture_map = &custom_material.textures[texture_map_id];
                    let txxf = material.texture_transform.unwrap_or_default();
                    let local_pixel_extents = find_pixel_extents(model, &txxf, texture_map);

                    match texture_extents.entry(*texture_map_id) {
                        Entry::Vacant(e) => {
                            e.insert(local_pixel_extents);
                        }
                        Entry::Occupied(mut e) => {
                            let v = e.get_mut();
                            *v = *v | local_pixel_extents;
                        }
                    }
                }
                if texture_extents.is_empty() {
                    continue;
                }
                for (id, bounds) in &texture_extents {
                    rects_to_place.push_rect(
                        *id,
                        None,
                        RectToInsert::new(bounds.width(), bounds.height(), 1),
                    );
                }
                let (size, layout) = pack_to_minimal_square(rects_to_place)?;

                // Remap UVs.
                for (part_id, material) in custom_material.parts.iter().enumerate() {
                    let Some(model) = custom_material.accessories.get_mut(&(part_id as u32)) else {
                        continue;
                    };
                    if model.vertices.is_empty() {
                        continue;
                    }
                    let Some(texture_map_id) = &material.texture_map else {
                        continue;
                    };
                    let extents = &texture_extents[texture_map_id];
                    let texture_map = &custom_material.textures[texture_map_id];
                    let location = &layout.packed_locations()[texture_map_id].1;

                    for vertex in &mut model.vertices {
                        let transformed = material
                            .texture_transform
                            .unwrap_or_default()
                            .transform_point(&vertex.map);
                        let original_source_pixel = point![
                            transformed.x * texture_map.width as f64,
                            transformed.y * texture_map.height as f64
                        ];
                        let cropped_source_pixel = point![
                            original_source_pixel.x - extents.min.x as f64,
                            original_source_pixel.y - extents.min.y as f64
                        ];
                        let dest_pixel = point![
                            cropped_source_pixel.x + location.x() as f64,
                            cropped_source_pixel.y + location.y() as f64
                        ];
                        let dest_uv =
                            point![dest_pixel.x / size as f64, dest_pixel.y / size as f64];
                        vertex.map = dest_uv;
                    }
                }

                // Generate atlases.
                let mut canvas = vec![0; size as usize * size as usize];
                for material in &custom_material.parts {
                    let Some(texture_map_id) = &material.texture_map else {
                        continue;
                    };
                    let Some((_, location)) = layout.packed_locations().get(texture_map_id) else {
                        continue;
                    };
                    let texture_map = &custom_material.textures[texture_map_id];

                    let extents = &texture_extents[texture_map_id];
                    blit(texture_map, extents, &mut canvas, size, location);
                }
                let texture_name = format!("{name}.{costume:03}.png");
                let writer = BufWriter::new(File::create(&texture_name)?);
                write_png(&canvas, size, size, writer)?;
            }
        } else {
            // Calculate layout.
            let mut rects_to_place = GroupedRectsToPlace::<_, u32>::new();
            for texture_id in texture_order {
                let bounds = &texture_extents[texture_id];
                rects_to_place.push_rect(
                    *texture_id,
                    None,
                    RectToInsert::new(bounds.width(), bounds.height(), 1),
                );
            }
            let (size, layout) = pack_to_minimal_square(rects_to_place)?;

            // Remap UVs.
            for (part, (_, cps)) in template.action_cells.cells[0]
                .parts
                .iter()
                .enumerate()
                .filter(|(i, _)| template.body_part_sets.groups[*i] == set)
                .enumerate()
            {
                let Some(model_id) = cps.model_id else {
                    continue;
                };
                let material = &first_custom_material.parts[part];
                let Some(texture_map_id) = &material.texture_map else {
                    continue;
                };
                let extents = &texture_extents[&texture_map_id];
                let texture_map = &first_custom_material.textures[texture_map_id];
                let location = &layout.packed_locations()[texture_map_id].1;
                for vertex in &mut template
                    .models
                    .get_mut(&(model_id as u32))
                    .unwrap()
                    .model
                    .vertices
                {
                    let transformed = material
                        .texture_transform
                        .unwrap_or_default()
                        .transform_point(&vertex.map);
                    let original_source_pixel = point![
                        transformed.x * texture_map.width as f64,
                        transformed.y * texture_map.height as f64
                    ];
                    let cropped_source_pixel = point![
                        original_source_pixel.x - extents.min.x as f64,
                        original_source_pixel.y - extents.min.y as f64
                    ];
                    let dest_pixel = point![
                        cropped_source_pixel.x + location.x() as f64,
                        cropped_source_pixel.y + location.y() as f64
                    ];
                    let dest_uv = point![dest_pixel.x / size as f64, dest_pixel.y / size as f64];
                    vertex.map = dest_uv;
                }
            }

            // Generate atlases.
            for &costume in &valid_costumes {
                let custom_material = &template.materials[&costume];
                let mut canvas = vec![0u8; size as usize * size as usize];
                let mut copied = HashSet::new();
                for (part, (_, cps)) in template.action_cells.cells[0]
                    .parts
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| template.body_part_sets.groups[*i] == set)
                    .enumerate()
                {
                    if cps.model_id.is_none() {
                        continue;
                    }
                    let material = &custom_material.parts[part];
                    let Some(texture_map) = &material.texture_map else {
                        continue;
                    };
                    let Some(first_texture_map_id) = first_custom_material.parts[part].texture_map else {
                        continue;
                    };
                    let texture_map = &custom_material.textures[texture_map];
                    if copied.insert(first_texture_map_id) {
                        let extents = &texture_extents[&first_texture_map_id];
                        let location = &layout.packed_locations()[&first_texture_map_id].1;
                        blit(texture_map, extents, &mut canvas, size, location);
                    }
                }

                let texture_name = format!("{name}.{costume:03}.png");
                let writer = BufWriter::new(File::create(&texture_name)?);
                write_png(&canvas, size, size, writer)?;
            }
        }

        // Clear transforms.
        for &costume in set_costumes {
            let material = template.materials.get_mut(&costume).unwrap();
            for part in &mut material.parts {
                part.texture_transform = None;
            }
        }
    }

    Ok(())
}

const PALETTE_BMP: &[u8] =
    include_bytes!("../../3DMMForever/src/building/bitmaps/palette/socbase.bmp");
lazy_static! {
    static ref PALETTE: Vec<u8> = {
        let bmp = RawBmp::from_slice(PALETTE_BMP).unwrap();
        let color_table = bmp.color_table().unwrap();
        let mut palette = Vec::with_capacity(256 * 3);
        for i in 0..color_table.len() {
            let color = color_table.get(i as u32).unwrap();
            palette.push(color.r());
            palette.push(color.g());
            palette.push(color.b());
        }
        palette
    };
}

fn export_model(name: &str, template: &TemplateData) -> Result<()> {
    let vrm_armature: Index<Node> = Index::new(0);
    let mut doc = Root {
        asset: Asset {
            generator: Some("3dmm-dump".to_owned()),
            ..Default::default()
        },
        scene: Some(Index::new(0)),
        scenes: vec![Scene {
            nodes: vec![vrm_armature],
            extensions: Default::default(),
            extras: Default::default(),
            name: Default::default(),
        }],
        nodes: vec![Node {
            name: Some("Armature".to_owned()),
            camera: Default::default(),
            children: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
            matrix: Default::default(),
            mesh: Default::default(),
            rotation: Default::default(),
            scale: Default::default(),
            translation: Default::default(),
            skin: Default::default(),
            weights: Default::default(),
        }],
        buffers: vec![Buffer {
            byte_length: 0,
            name: Default::default(),
            uri: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }],
        samplers: vec![Sampler {
            mag_filter: Some(Checked::Valid(MagFilter::Nearest)),
            min_filter: Some(Checked::Valid(MinFilter::Nearest)),
            wrap_s: Checked::Valid(WrappingMode::ClampToEdge),
            wrap_t: Checked::Valid(WrappingMode::ClampToEdge),
            name: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        }],
        ..Default::default()
    };
    let mut buffer = Vec::new();

    let mut armature_nodes = Vec::with_capacity(template.armature.parents.len());
    for index in 0..template.armature.parents.len() {
        let scale_node = Index::new(doc.nodes.len() as u32);
        let rotate_node = Index::new(doc.nodes.len() as u32 + 1);
        let translate_node = Index::new(doc.nodes.len() as u32 + 2);
        armature_nodes.push((scale_node, translate_node));
        let matrix_id = template.action_cells.cells[0].parts[index].matrix_id;
        let matrix = template.action_transforms.transforms[matrix_id as usize].into_inner();
        let (translation, rotation, scale) = decompose_cps_transform(matrix);

        doc.nodes.push(Node {
            name: Some(format!("node.{index:03}.scale")),
            // rotation: Some(UnitQuaternion([
            //     isometry.rotation[0] as f32,
            //     isometry.rotation[1] as f32,
            //     isometry.rotation[2] as f32,
            //     isometry.rotation[3] as f32,
            // ])),
            rotation: None,
            scale: Some([scale.x as f32, scale.y as f32, scale.z as f32]),
            // translation: Some([
            //     translation.x as f32,
            //     translation.y as f32,
            //     translation.z as f32,
            // ]),
            translation: None,
            camera: Default::default(),
            children: Some(vec![rotate_node]),
            extensions: Default::default(),
            extras: Default::default(),
            matrix: Default::default(),
            mesh: Default::default(),
            skin: Default::default(),
            weights: Default::default(),
        });
        doc.nodes.push(Node {
            name: Some(format!("node.{index:03}.rotate")),
            rotation: Some(UnitQuaternion([
                rotation[0] as f32,
                rotation[1] as f32,
                rotation[2] as f32,
                rotation[3] as f32,
            ])),
            scale: None,
            translation: None,
            camera: Default::default(),
            children: Some(vec![translate_node]),
            extensions: Default::default(),
            extras: Default::default(),
            matrix: Default::default(),
            mesh: Default::default(),
            skin: Default::default(),
            weights: Default::default(),
        });
        doc.nodes.push(Node {
            name: Some(format!("node.{index:03}.translate")),
            rotation: None,
            scale: None,
            translation: Some([
                translation.x as f32,
                translation.y as f32,
                translation.z as f32,
            ]),
            camera: Default::default(),
            children: None,
            extensions: Default::default(),
            extras: Default::default(),
            matrix: Default::default(),
            mesh: Default::default(),
            skin: Default::default(),
            weights: Default::default(),
        });
    }

    for (index, parent_index) in template.armature.parents.iter().copied().enumerate() {
        let node = armature_nodes[index];
        const PARENT_ROOT: u16 = 65535;
        let parent_node = match parent_index {
            PARENT_ROOT => Some(vrm_armature),
            o => armature_nodes.get(o as usize).map(|n| n.1),
        };
        let Some(parent_node) = parent_node else {
            bail!("parent index out of range");
        };
        doc.get_mut(parent_node)
            .unwrap()
            .children
            .get_or_insert(Vec::new())
            .push(node.0);
    }

    let mut texture_materials =
        HashMap::with_capacity(template.costumes.part_sets.iter().flatten().count());
    for &id in template.costumes.part_sets.iter().flatten() {
        let image_index = Index::new(doc.images.len() as u32);
        doc.images.push(Image {
            name: Some(format!("tmap.{id:03}.image")),
            uri: Some(format!("{name}.{id:03}.png")),
            buffer_view: Default::default(),
            mime_type: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let texture_index = Index::<Texture>::new(doc.textures.len() as u32);
        doc.textures.push(Texture {
            name: Some(format!("tmap.{id:03}")),
            sampler: Some(Index::new(0)),
            source: image_index,
            extensions: Default::default(),
            extras: Default::default(),
        });

        let material_index = Index::<Material>::new(doc.materials.len() as u32);
        doc.materials.push(Material {
            name: Some(format!("material.{id:03}")),
            alpha_mode: Checked::Valid(AlphaMode::Opaque),
            pbr_metallic_roughness: PbrMetallicRoughness {
                base_color_texture: Some(Info {
                    index: texture_index,
                    tex_coord: 0,
                    extensions: Default::default(),
                    extras: Default::default(),
                }),
                base_color_factor: PbrBaseColorFactor([1.0; 4]),
                metallic_factor: Default::default(),
                roughness_factor: Default::default(),
                metallic_roughness_texture: Default::default(),
                extensions: Default::default(),
                extras: Default::default(),
            },
            alpha_cutoff: Default::default(),
            double_sided: Default::default(),
            emissive_factor: Default::default(),
            normal_texture: Default::default(),
            occlusion_texture: Default::default(),
            emissive_texture: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        });

        texture_materials.insert(id, material_index);
    }

    let mut materials = HashMap::new();
    for (index, set_materials) in template.materials.iter() {
        for (part_index, material_data) in set_materials.parts.iter().enumerate() {
            let material_index = if material_data.texture_map.is_some() {
                texture_materials[index]
            } else {
                let material_index = Index::<Material>::new(doc.materials.len() as u32);
                doc.materials.push(Material {
                    name: Some(format!("material.{index:03}.{part_index:03}")),
                    alpha_mode: Checked::Valid(AlphaMode::Opaque),
                    pbr_metallic_roughness: PbrMetallicRoughness {
                        base_color_texture: None,
                        base_color_factor: PbrBaseColorFactor([
                            PALETTE[material_data.material.color as usize * 3] as f32 / 255.0,
                            PALETTE[material_data.material.color as usize * 3 + 1] as f32 / 255.0,
                            PALETTE[material_data.material.color as usize * 3 + 2] as f32 / 255.0,
                            1.0,
                        ]),
                        metallic_factor: Default::default(),
                        roughness_factor: Default::default(),
                        metallic_roughness_texture: Default::default(),
                        extensions: Default::default(),
                        extras: Default::default(),
                    },
                    emissive_factor: EmissiveFactor([material_data.material.ambient; 3]),
                    alpha_cutoff: Default::default(),
                    double_sided: Default::default(),
                    normal_texture: Default::default(),
                    occlusion_texture: Default::default(),
                    emissive_texture: Default::default(),
                    extensions: Default::default(),
                    extras: Default::default(),
                });

                material_index
            };

            materials.insert((*index, part_index as u32), material_index);
        }
    }

    for (index, cps) in template.action_cells.cells[0].parts.iter().enumerate() {
        let set = template.body_part_sets.groups[index];
        let part_index = template
            .body_part_sets
            .groups
            .iter()
            .take(index)
            .filter(|s| **s == set)
            .count() as u32;
        let material_set = template.costumes.part_sets[set as usize][0];
        let material_index = materials[&(material_set, part_index)];

        let Some(model) = template.materials[&material_set].accessories.get(&part_index).or_else(|| cps.model_id.and_then(|model_id| template.models.get(&(model_id as u32))).map(|m| &m.model)) else {
            continue;
        };
        let Some(armature_node) = armature_nodes.get(index).copied() else {
            bail!("node index out of range");
        };

        if model.faces.is_empty() {
            continue;
        }

        let position_offset = buffer.len();

        for vertex in model.vertices.iter() {
            buffer.write_f32::<LittleEndian>(vertex.position.x as f32)?;
            buffer.write_f32::<LittleEndian>(vertex.position.y as f32)?;
            buffer.write_f32::<LittleEndian>(vertex.position.z as f32)?;
        }

        let position_buffer = Index::new(doc.buffer_views.len() as u32);
        doc.buffer_views.push(View {
            buffer: Index::new(0),
            byte_length: (buffer.len() - position_offset) as u32,
            byte_offset: Some(position_offset as u32),
            byte_stride: Some((3 * mem::size_of::<f32>()) as u32),
            name: Some(format!("mesh.{index:03}.positions")),
            target: Some(Checked::Valid(Target::ArrayBuffer)),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let position = Index::new(doc.accessors.len() as u32);
        doc.accessors.push(Accessor {
            buffer_view: Some(position_buffer),
            component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
            count: model.vertices.len() as u32,
            min: Some(Value::Array(vec![
                Value::Number(Number::from_f64(model.bounds.min.x).unwrap()),
                Value::Number(Number::from_f64(model.bounds.min.y).unwrap()),
                Value::Number(Number::from_f64(model.bounds.min.z).unwrap()),
            ])),
            max: Some(Value::Array(vec![
                Value::Number(Number::from_f64(model.bounds.max.x).unwrap()),
                Value::Number(Number::from_f64(model.bounds.max.y).unwrap()),
                Value::Number(Number::from_f64(model.bounds.max.z).unwrap()),
            ])),
            name: Some(format!("mesh.{index:03}.positions")),
            type_: Checked::Valid(Type::Vec3),
            byte_offset: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
            normalized: Default::default(),
            sparse: Default::default(),
        });

        let normal_offset = buffer.len();

        for vertex in model.vertices.iter() {
            buffer.write_f32::<LittleEndian>(vertex.normal.x)?;
            buffer.write_f32::<LittleEndian>(vertex.normal.y)?;
            buffer.write_f32::<LittleEndian>(vertex.normal.z)?;
        }

        let normal_buffer = Index::new(doc.buffer_views.len() as u32);
        doc.buffer_views.push(View {
            buffer: Index::new(0),
            byte_length: (buffer.len() - normal_offset) as u32,
            byte_offset: Some(normal_offset as u32),
            byte_stride: Some((3 * mem::size_of::<f32>()) as u32),
            name: Some(format!("mesh.{index:03}.normals")),
            target: Some(Checked::Valid(Target::ArrayBuffer)),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let normal = Index::new(doc.accessors.len() as u32);
        doc.accessors.push(Accessor {
            buffer_view: Some(normal_buffer),
            component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
            count: model.vertices.len() as u32,
            name: Some(format!("mesh.{index:03}.normals")),
            type_: Checked::Valid(Type::Vec3),
            byte_offset: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
            min: Default::default(),
            max: Default::default(),
            normalized: Default::default(),
            sparse: Default::default(),
        });

        let texcoord_offset = buffer.len();

        for vertex in model.vertices.iter() {
            let uv = vertex.map;
            buffer.write_f32::<LittleEndian>(uv.x as f32)?;
            buffer.write_f32::<LittleEndian>(uv.y as f32)?;
        }

        let texcoord_buffer = Index::new(doc.buffer_views.len() as u32);
        doc.buffer_views.push(View {
            buffer: Index::new(0),
            byte_length: (buffer.len() - texcoord_offset) as u32,
            byte_offset: Some(texcoord_offset as u32),
            byte_stride: Some((2 * mem::size_of::<f32>()) as u32),
            name: Some(format!("mesh.{index:03}.texcoords")),
            target: Some(Checked::Valid(Target::ArrayBuffer)),
            extensions: Default::default(),
            extras: Default::default(),
        });

        let texcoord = Index::new(doc.accessors.len() as u32);
        doc.accessors.push(Accessor {
            buffer_view: Some(texcoord_buffer),
            component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
            count: model.vertices.len() as u32,
            name: Some(format!("mesh.{index:03}.texcoords")),
            type_: Checked::Valid(Type::Vec2),
            byte_offset: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
            min: Default::default(),
            max: Default::default(),
            normalized: Default::default(),
            sparse: Default::default(),
        });

        let index_offset = buffer.len();

        for face in model.faces.iter() {
            for vertex in &face.vertices {
                buffer.write_u16::<LittleEndian>(*vertex)?;
            }
        }

        let index_buffer = Index::new(doc.buffer_views.len() as u32);
        doc.buffer_views.push(View {
            buffer: Index::new(0),
            byte_length: (buffer.len() - index_offset) as u32,
            byte_offset: Some(index_offset as u32),
            name: Some(format!("mesh.{index:03}.indices")),
            target: Some(Checked::Valid(Target::ElementArrayBuffer)),
            byte_stride: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
        });

        // the index buffer length might not be a multiple of four but the next buffer must start at a four byte alignment.
        buffer.extend(iter::repeat(0).take(3 - (buffer.len() + 3) % 4));

        let indices = Index::new(doc.accessors.len() as u32);
        doc.accessors.push(Accessor {
            buffer_view: Some(index_buffer),
            component_type: Checked::Valid(GenericComponentType(ComponentType::U16)),
            count: model.faces.len() as u32 * 3,
            name: Some(format!("mesh.{index:03}.indices")),
            type_: Checked::Valid(Type::Scalar),
            byte_offset: Default::default(),
            extensions: Default::default(),
            extras: Default::default(),
            min: Default::default(),
            max: Default::default(),
            normalized: Default::default(),
            sparse: Default::default(),
        });

        let mesh_index = Index::new(doc.meshes.len() as u32);
        doc.get_mut(armature_node.1).unwrap().mesh = Some(mesh_index);
        let mesh = Mesh {
            name: Some(format!("mesh.{index:03}")),
            primitives: vec![Primitive {
                attributes: hashmap! {
                    Checked::Valid(Semantic::Positions) => position,
                    Checked::Valid(Semantic::Normals) => normal,
                    Checked::Valid(Semantic::TexCoords(0)) => texcoord,
                },
                indices: Some(indices),
                extensions: Default::default(),
                extras: Default::default(),
                material: Some(material_index),
                mode: Checked::Valid(Mode::Triangles),
                targets: Default::default(),
            }],
            extensions: Default::default(),
            extras: Default::default(),
            weights: Default::default(),
        };
        doc.meshes.push(mesh)
    }

    doc.buffers[0].byte_length = buffer.len() as u32;

    let f = BufWriter::new(File::create(format!("{name}.glb"))?);
    Glb {
        header: Header {
            magic: Default::default(),
            version: Default::default(),
            length: Default::default(),
        },
        json: Cow::Owned(doc.to_vec()?),
        bin: Some(Cow::Owned(buffer)),
    }
    .to_writer(f)?;
    Ok(())
}

fn decompose_cps_transform(
    matrix: Matrix4<f64>,
) -> (
    Translation3<f64>,
    nalgebra::UnitQuaternion<f64>,
    Scale3<f64>,
) {
    let scale = Scale3::new(
        matrix.fixed_view::<1, 3>(0, 0).magnitude(),
        matrix.fixed_view::<1, 3>(1, 0).magnitude(),
        matrix.fixed_view::<1, 3>(2, 0).magnitude(),
    );
    let matrix = scale.pseudo_inverse().to_homogeneous() * matrix;

    let rotation: nalgebra::UnitQuaternion<f64> = nalgebra::convert_unchecked(matrix);
    let matrix = rotation.inverse().to_homogeneous() * matrix;

    let translation = Translation3::from(Matrix::from(matrix.fixed_view::<3, 1>(0, 3)));

    (translation, rotation, scale)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix4, Scale3, Translation3, UnitQuaternion, Vector3};

    use crate::decompose_cps_transform;

    #[test]
    fn decompose_transform() {
        const TOLERANCE: f64 = 0.000_001;
        let original_translation = Translation3::new(3.0, 5.0, 7.0);
        let original_translated = original_translation.to_homogeneous() * Matrix4::identity();

        let original_rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.125)
            * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.25)
            * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5);
        let original_rotated = original_rotation.to_homogeneous() * original_translated;

        let original_scale = Scale3::new(11.0, 13.0, 17.0);
        let original_transform = original_scale.to_homogeneous() * original_rotated;

        let (translation, rotation, scale) = decompose_cps_transform(original_transform);

        assert!(
            (scale.vector - original_scale.vector).amax() < TOLERANCE,
            "{original_scale} != {scale}",
        );

        assert!(
            (rotation.vector() - original_rotation.vector()).amax() < TOLERANCE,
            "{original_rotation} != {rotation}",
        );

        assert!(
            (translation.vector - original_translation.vector).amax() < TOLERANCE,
            "{original_translation} != {translation}",
        );
    }
}

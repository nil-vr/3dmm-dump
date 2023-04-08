use std::mem;

use anyhow::{bail, ensure, Result};
use byteorder::ByteOrder;
use nalgebra::{point, vector, Point2, Point3, Vector3};
use rgb::RGB8;
use zerocopy::{FromBytes, U16, U32};

use crate::{
    brender::{Fraction, Scalar},
    order::Loader,
};

#[derive(FromBytes)]
#[repr(C)]
pub struct ModelOnFile<O>
where
    O: ByteOrder,
{
    byte_order: U16<O>,
    _osk: U16<O>,
    vertex_count: U16<O>,
    face_count: U16<O>,
    radius: Scalar<O>,
    bounds: BoundsOnFile<O>,
    pivot: Vector3OnFile<O>,
}

#[derive(FromBytes)]
#[repr(C)]
pub struct BoundsOnFile<O>
where
    O: ByteOrder,
{
    min: Vector3OnFile<O>,
    max: Vector3OnFile<O>,
}

#[derive(FromBytes)]
#[repr(C)]
pub struct Vector2OnFile<O>
where
    O: ByteOrder,
{
    v: [Scalar<O>; 2],
}

#[derive(FromBytes)]
#[repr(C)]
pub struct Vector3OnFile<O>
where
    O: ByteOrder,
{
    v: [Scalar<O>; 3],
}

#[derive(FromBytes)]
#[repr(C)]
pub struct FVector3OnFile<O>
where
    O: ByteOrder,
{
    v: [Fraction<O>; 3],
}

#[derive(FromBytes)]
#[repr(C)]
pub struct VertexOnFile<O>
where
    O: ByteOrder,
{
    point: Vector3OnFile<O>,
    map: Vector2OnFile<O>,
    index: u8,
    red: u8,
    green: u8,
    blue: u8,
    _reserved: U16<O>,
    normal: FVector3OnFile<O>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Bounds {
    pub min: Point3<f64>,
    pub max: Point3<f64>,
}

impl<O> From<BoundsOnFile<O>> for Bounds
where
    O: ByteOrder,
{
    fn from(value: BoundsOnFile<O>) -> Self {
        Bounds {
            min: value.min.into(),
            max: value.max.into(),
        }
    }
}

impl<O> From<Vector3OnFile<O>> for Point3<f64>
where
    O: ByteOrder,
{
    fn from(value: Vector3OnFile<O>) -> Self {
        point![value.v[0].into(), value.v[1].into(), value.v[2].into()]
    }
}

impl<O> From<FVector3OnFile<O>> for Vector3<f32>
where
    O: ByteOrder,
{
    fn from(value: FVector3OnFile<O>) -> Self {
        vector![value.v[0].into(), value.v[1].into(), value.v[2].into()]
    }
}

impl<O> From<Vector2OnFile<O>> for Point2<f64>
where
    O: ByteOrder,
{
    fn from(value: Vector2OnFile<O>) -> Self {
        point![value.v[0].into(), value.v[1].into()]
    }
}

#[derive(Debug)]
pub struct Vertex {
    pub position: Point3<f64>,
    pub map: Point2<f64>,
    pub index: u8,
    pub color: RGB8,
    pub normal: Vector3<f32>,
}

impl<O> From<VertexOnFile<O>> for Vertex
where
    O: ByteOrder,
{
    fn from(value: VertexOnFile<O>) -> Self {
        Vertex {
            position: value.point.into(),
            map: value.map.into(),
            index: value.index,
            color: RGB8::new(value.red, value.green, value.blue),
            normal: value.normal.into(),
        }
    }
}

#[derive(Debug)]
pub struct Model {
    pub _radius: f64,
    pub bounds: Bounds,
    pub _pivot: Point3<f64>,
    pub vertices: Vec<Vertex>,
    pub faces: Vec<Face>,
}

#[derive(FromBytes)]
#[repr(C)]
pub struct FaceOnFile<O>
where
    O: ByteOrder,
{
    vertices: [U16<O>; 3],
    edges: [U16<O>; 3],
    material: U32<O>,
    smoothing: U16<O>,
    flags: u8,
    _pad0: u8,
    normal: FVector3OnFile<O>,
    d: Scalar<O>,
    _pad1: u16,
}

#[derive(Debug)]
pub struct Face {
    pub vertices: [u16; 3],
    pub edges: [u16; 3],
    pub material: u32,
    pub smoothing: u16,
    pub flags: u8,
    pub normal: Vector3<f32>,
    pub d: f64,
}

impl<O> From<FaceOnFile<O>> for Face
where
    O: ByteOrder,
{
    fn from(value: FaceOnFile<O>) -> Self {
        Face {
            vertices: value.vertices.map(|v| v.get()),
            edges: value.edges.map(|v| v.get()),
            material: value.material.get(),
            smoothing: value.smoothing.get(),
            flags: value.flags,
            normal: value.normal.into(),
            d: value.d.into(),
        }
    }
}

struct GroupBy<'a, T, F>(&'a [T], F)
where
    T: 'a,
    F: Fn(&'a T, &'a T) -> bool;

impl<'a, T, F> Iterator for GroupBy<'a, T, F>
where
    T: 'a,
    F: Fn(&'a T, &'a T) -> bool,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            return None;
        }
        for i in 1..self.0.len() {
            if !(self.1)(&self.0[i - 1], &self.0[i]) {
                let (group, rest) = self.0.split_at(i);
                self.0 = rest;
                return Some(group);
            }
        }
        Some(mem::take(&mut self.0))
    }
}

impl<'a> Loader<'a> for Model {
    type OnFile<O> = ModelOnFile<O>
    where
        O: ByteOrder;

    fn byte_order<O>(on_file: &Self::OnFile<O>) -> u16
    where
        O: ByteOrder,
    {
        on_file.byte_order.get()
    }

    fn into_native<O>(data: Self::OnFile<O>, full_input: &'a [u8]) -> Result<Self>
    where
        O: ByteOrder,
    {
        let mut input = &full_input[mem::size_of::<ModelOnFile<O>>()..];

        let bounds: Bounds = data.bounds.into();
        let mut bounds = if bounds == Default::default() {
            None
        } else {
            Some(bounds)
        };
        let mut vertices = Vec::with_capacity(data.vertex_count.get() as usize);
        for _ in 0..data.vertex_count.get() {
            let Some(vertex) = VertexOnFile::<O>::read_from_prefix(input) else {
                bail!("EOF in vertices");
            };
            input = &input[mem::size_of::<VertexOnFile<O>>()..];
            let vertex: Vertex = vertex.into();
            if let Some(bounds) = &mut bounds {
                bounds.min.x = bounds.min.x.min(vertex.position.x);
                bounds.min.y = bounds.min.y.min(vertex.position.y);
                bounds.min.z = bounds.min.z.min(vertex.position.z);
                bounds.max.x = bounds.max.x.max(vertex.position.x);
                bounds.max.y = bounds.max.y.max(vertex.position.y);
                bounds.max.z = bounds.max.z.max(vertex.position.z);
            } else {
                bounds = Some(Bounds {
                    min: vertex.position,
                    max: vertex.position,
                });
            }
            vertices.push(vertex);
        }

        let mut faces: Vec<Face> = Vec::with_capacity(data.face_count.get() as usize);
        for _ in 0..data.face_count.get() {
            let Some(face) = FaceOnFile::<O>::read_from_prefix(input) else {
                bail!("EOF in faces");
            };
            input = &input[mem::size_of::<FaceOnFile<O>>()..];
            faces.push(face.into());
        }

        ensure!(input.is_empty(), "Did not read complete model");

        // Port of BRender 1.3.2 prepmesh code for normal calculation.
        // sitobren is supposed to do this, but I can't go back in time 28 years to fix it.
        if data.radius.is_zero() && !faces.is_empty() {
            struct PrepVertex {
                vertex: usize,
                face: usize,
            }

            for face in &mut faces {
                // calculate face normals
                let v = [
                    vertices[face.vertices[0] as usize].position,
                    vertices[face.vertices[1] as usize].position,
                    vertices[face.vertices[2] as usize].position,
                ];
                let a = v[0] - v[1];
                let b = vertices[face.vertices[2] as usize].position
                    - vertices[face.vertices[0] as usize].position;

                face.normal = vector![
                    (a.y * b.z - a.z * b.y) as f32,
                    (a.z * b.x - a.x * b.z) as f32,
                    (a.x * b.y - a.y * b.x) as f32
                ];
                if face.normal.magnitude_squared() < 0.0001 {
                    face.normal = vector![0.0, 0.0, 1.0];
                } else {
                    face.normal = face.normal.normalize();
                }
                face.normal *= -1.0;
                assert!(!face.normal[0].is_nan());
                assert!(!face.normal[1].is_nan());
                assert!(!face.normal[2].is_nan());

                face.d = face.normal.x as f64 * v[0].x
                    + face.normal.y as f64 * v[0].y
                    + face.normal.z as f64 * v[0].z;

                // 0 means all groups
                if face.smoothing == 0 {
                    face.smoothing = u16::MAX;
                }
            }

            let temp_verts: Vec<_> = faces
                .iter()
                .enumerate()
                .flat_map(|(face_index, face)| {
                    face.vertices.iter().map(move |v| PrepVertex {
                        vertex: *v as usize,
                        face: face_index,
                    })
                })
                .collect();

            let mut normals = vec![Vector3::default(); vertices.len()];

            let vertex_compare_smoothing = |a: &PrepVertex, b: &PrepVertex| {
                let a = vertices[a.vertex].position;
                let b = vertices[b.vertex].position;
                a.x.partial_cmp(&b.x)
                    .unwrap()
                    .then_with(|| a.y.partial_cmp(&b.y).unwrap())
                    .then_with(|| a.z.partial_cmp(&b.z).unwrap())
            };

            let mut sorted_faces: Vec<usize> = (0..faces.len()).collect();
            sorted_faces.sort_unstable_by(|a, b| faces[*a].material.cmp(&faces[*b].material));

            let mut sorted_vertices: Vec<usize> = (0..temp_verts.len()).collect();
            sorted_vertices.sort_unstable_by(|a, b| {
                vertex_compare_smoothing(&temp_verts[*a], &temp_verts[*b])
            });

            for weld in GroupBy(&sorted_vertices, |a, b| {
                vertex_compare_smoothing(
                    &temp_verts[sorted_vertices[*a]],
                    &temp_verts[sorted_vertices[*b]],
                )
                .is_eq()
            }) {
                for i in 0..weld.len() {
                    for j in 0..weld.len() {
                        let vertex = &temp_verts[weld[i]];
                        let j = &faces[temp_verts[weld[j]].face];
                        if (faces[vertex.face].smoothing & j.smoothing) != 0 {
                            let normal = &mut normals[temp_verts[weld[i]].vertex];
                            *normal += j.normal;
                        }
                    }
                }
            }

            for (vertex, normal) in vertices.iter_mut().zip(normals) {
                if normal.magnitude_squared() < 0.0001 {
                    vertex.normal = vector![0.0, 0.0, -1.0];
                } else {
                    vertex.normal = normal.normalize();
                }
            }
        }

        Ok(Model {
            _radius: data.radius.into(),
            bounds: bounds.unwrap_or_default(),
            _pivot: data.pivot.into(),
            vertices,
            faces,
        })
    }
}

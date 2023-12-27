use crate::extensions;
use serde::{Deserialize, Serialize};
use serde_repr::*;
use std::collections::HashMap;

use serde_json::Value;

#[derive(Serialize_repr, Deserialize_repr, Debug, PartialEq, Eq, Clone, Copy, Default)]
#[repr(u8)]
pub enum PrimitiveMode {
    Points = 0,
    Lines = 1,
    LineLoop = 2,
    LineStrip = 3,
    #[default]
    Triangles = 4,
    TriangleStrip = 5,
    TriangleFan = 6,
}

/// Geometry to be rendered with the given material.
#[derive(Serialize, Deserialize, Debug, Default)]
#[serde[rename_all = "camelCase"]]
#[serde(deny_unknown_fields)]
pub struct MeshPrimitive {
    /// A plain JSON object, where each key corresponds to a mesh attribute semantic and each value is the index of the accessor containing attribute's data.
    pub attributes: HashMap<String, u32>, // required

    /// The index of the accessor that contains the vertex indices.  When this is undefined, the primitive defines non-indexed geometry.  When defined, the accessor **MUST** have `SCALAR` type and an unsigned integer component type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indices: Option<u32>,

    /// The index of the material to apply to this primitive when rendering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<u32>,

    /// The topology type of primitives to render.
    #[serde(default)]
    pub mode: PrimitiveMode,

    /// An array of morph targets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<HashMap<String, u32>>>,

    /// JSON object with extension-specific objects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<extensions::mesh::MeshPrimitive>,

    /// Application-specific data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

impl MeshPrimitive {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}

/// A set of primitives to be rendered.  Its global transform is defined by a node that references it.
#[derive(Serialize, Deserialize, Debug, Default)]
#[serde[rename_all = "camelCase"]]
#[serde(deny_unknown_fields)]
pub struct Mesh {
    /// The user-defined name of this object.  This is not necessarily unique, e.g., an accessor and a buffer could have the same name, or two accessors could even have the same name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// An array of primitives, each defining geometry to be rendered.
    pub primitives: Vec<MeshPrimitive>,

    /// Array of weights to be applied to the morph targets. The number of array elements **MUST** match the number of morph targets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f32>>,

    /// JSON object with extension-specific objects.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<MeshExtensions>,

    /// Application-specific data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct MeshExtensions {
    #[serde(flatten)]
    others: HashMap<String, Value>,
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}
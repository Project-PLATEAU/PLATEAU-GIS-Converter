use std::ops::{BitAnd, BitAndAssign, BitOrAssign};

use nusamai_citygml::{
    object::{ObjectStereotype, Value},
    schema::Schema,
};
use nusamai_plateau::Entity;

use crate::{pipeline::Feedback, transformer::Transform};

#[derive(Clone, Copy)]
pub enum LodFilterMode {
    Highest,
    Lowest,
    TexturedHighest,
    All,
}

#[derive()]
pub struct FilterLodTransform {
    mask: LodMask,
    mode: LodFilterMode,
}

impl FilterLodTransform {
    pub fn new(mask: LodMask, mode: LodFilterMode) -> Self {
        Self { mask, mode }
    }
}

/// Transform to filter and split the LODs
impl Transform for FilterLodTransform {
    fn transform(&mut self, _feedback: &Feedback, mut entity: Entity, out: &mut Vec<Entity>) {
        // Extract the largest LOD with a texture. If there is no texture, extract the largest LOD.
        match self.mode {
            LodFilterMode::TexturedHighest => {
                // Pick the highest LOD that actually has a texture; fall back to the highest LOD.
                //
                // FIX: the previous implementation called the DESTRUCTIVE `edit_tree` inside the
                // probing loop and judged "has texture" from the whole-entity `appearance_store`
                // (LOD-independent, and never pruned by `edit_tree`). For entities without any
                // texture this progressively destroyed the geometry tree, making whole (untextured)
                // buildings disappear. We now probe NON-destructively using `polygon_textures`
                // (per-polygon texture assignment, already resolved by ApplyAppearanceTransform,
                // which runs before this transform) and call `edit_tree` exactly once.
                let available_lods = find_lods(&entity.root) & self.mask;
                let Some(highest_available_lod) = available_lods.highest_lod() else {
                    return;
                };
                let target_lod = {
                    let geom = entity.geometry_store.read().unwrap();
                    let mut chosen = None;
                    for lod in (0..=highest_available_lod).rev() {
                        if available_lods.has_lod(lod)
                            && lod_has_texture(&entity.root, lod, &geom.polygon_textures)
                        {
                            chosen = Some(lod);
                            break;
                        }
                    }
                    chosen.unwrap_or(highest_available_lod)
                };
                edit_tree(&mut entity.root, target_lod);
                out.push(entity);
            }
            LodFilterMode::Highest => {
                let lods = find_lods(&entity.root) & self.mask;
                let target_lod = lods.highest_lod();

                if let Some(target_lod) = target_lod {
                    edit_tree(&mut entity.root, target_lod);
                    out.push(entity);
                }
            }
            LodFilterMode::Lowest => {
                let lods = find_lods(&entity.root) & self.mask;
                let target_lod = lods.lowest_lod();

                if let Some(target_lod) = target_lod {
                    edit_tree(&mut entity.root, target_lod);
                    out.push(entity);
                }
            }
            LodFilterMode::All => {
                out.push(entity);
            }
        }
    }

    fn transform_schema(&self, _schema: &mut Schema) {
        // do nothing
    }
}

fn edit_tree(value: &mut Value, target_lod: u8) -> bool {
    match value {
        Value::Object(obj) => {
            let mut retain = false;
            if let ObjectStereotype::Feature { geometries, .. } = &mut obj.stereotype {
                geometries.retain(|geom| geom.lod == target_lod);
                retain |= !geometries.is_empty();
            } else {
                // Data or Object Stereotype
                retain = true;
            }
            obj.attributes.retain(|_, value| {
                let retain_child = edit_tree(value, target_lod);
                retain |= retain_child;
                retain_child
            });
            retain
        }
        Value::Array(arr) => {
            arr.retain_mut(|value| edit_tree(value, target_lod));
            !arr.is_empty()
        }
        _ => true,
    }
}

/// Whether any geometry at `target_lod` has a textured polygon (non-destructive).
///
/// `GeometryRef { pos, len }` spans `[pos, pos+len)` of the polygons, aligned with
/// `polygon_textures` (per-polygon texture assignment). `polygon_textures` is empty when
/// appearance resolution is disabled, in which case this returns false (callers fall back
/// to the highest LOD).
fn lod_has_texture(value: &Value, target_lod: u8, polygon_textures: &[Option<u32>]) -> bool {
    match value {
        Value::Object(obj) => {
            if let ObjectStereotype::Feature { geometries, .. } = &obj.stereotype {
                for geom in geometries {
                    if geom.lod == target_lod {
                        let start = geom.pos as usize;
                        let end = start + geom.len as usize;
                        if polygon_textures
                            .get(start..end)
                            .is_some_and(|s| s.iter().any(|t| t.is_some()))
                        {
                            return true;
                        }
                    }
                }
            }
            obj.attributes
                .values()
                .any(|v| lod_has_texture(v, target_lod, polygon_textures))
        }
        Value::Array(arr) => arr
            .iter()
            .any(|v| lod_has_texture(v, target_lod, polygon_textures)),
        _ => false,
    }
}

fn find_lods(value: &Value) -> LodMask {
    let mut mask = LodMask::default();
    match value {
        Value::Object(obj) => {
            if let ObjectStereotype::Feature { geometries, .. } = &obj.stereotype {
                geometries.iter().for_each(|geom| mask.add_lod(geom.lod));
            }
            for value in obj.attributes.values() {
                mask |= find_lods(value);
            }
        }
        Value::Array(arr) => {
            arr.iter().for_each(|value| mask |= find_lods(value));
        }
        _ => {}
    }
    mask
}

#[derive(Default, Clone, Copy)]
pub struct LodMask(
    u8, // lods bit mask
);

impl LodMask {
    pub fn all() -> Self {
        Self(0b11111)
    }

    pub fn add_lod(&mut self, lod_no: u8) {
        self.0 |= 1 << lod_no;
    }

    pub fn remove_lod(&mut self, lod_no: u8) {
        self.0 |= 1 << lod_no;
    }

    pub fn has_lod(&self, lod_no: u8) -> bool {
        self.0 & (1 << lod_no) != 0
    }

    /// Returns the highest LOD number.
    ///
    /// It returns `None` if none of the LODs are set.
    pub fn highest_lod(&self) -> Option<u8> {
        match self.0 {
            0 => None,
            _ => Some(7 - self.0.leading_zeros() as u8),
        }
    }

    /// Returns the lowest LOD number.
    ///
    /// It returns `None` if none of the LODs are set.
    pub fn lowest_lod(&self) -> Option<u8> {
        match self.0 {
            0 => None,
            _ => Some(self.0.trailing_zeros() as u8),
        }
    }
}

impl BitOrAssign for LodMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAndAssign for LodMask {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitAnd for LodMask {
    type Output = LodMask;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::RwLock;

    use nusamai_citygml::{
        geometry::{GeometryRef, GeometryType},
        object::Object,
        GeometryStore,
    };
    use nusamai_plateau::Entity;

    use super::*;
    use crate::pipeline::feedback::watcher;

    /// Build a single-feature entity from `(lod, polygon_index)` geometries and per-polygon
    /// texture assignments, run `TexturedHighest`, and return the surviving geometry LODs.
    fn run_textured_highest(geoms: &[(u8, u32)], polygon_textures: Vec<Option<u32>>) -> Vec<u8> {
        let geometries: Vec<GeometryRef> = geoms
            .iter()
            .map(|&(lod, pos)| GeometryRef {
                ty: GeometryType::Solid,
                lod,
                pos,
                len: 1,
            })
            .collect();
        let geometry_store = GeometryStore {
            polygon_textures,
            ..Default::default()
        };
        let entity = Entity {
            root: Value::Object(Object {
                typename: "bldg:Building".into(),
                attributes: Default::default(),
                stereotype: ObjectStereotype::Feature {
                    id: "b1".into(),
                    geometries,
                },
            }),
            base_url: url::Url::parse("file:///dummy").unwrap(),
            geometry_store: RwLock::new(geometry_store).into(),
            appearance_store: Default::default(),
        };
        let (_watcher, feedback, _canceller) = watcher();
        let mut transform = FilterLodTransform::new(LodMask::all(), LodFilterMode::TexturedHighest);
        let mut out = Vec::new();
        transform.transform(&feedback, entity, &mut out);
        let mut lods = Vec::new();
        for entity in &out {
            if let Value::Object(obj) = &entity.root {
                if let ObjectStereotype::Feature { geometries, .. } = &obj.stereotype {
                    lods.extend(geometries.iter().map(|g| g.lod));
                }
            }
        }
        lods.sort_unstable();
        lods
    }

    #[test]
    fn textured_highest_keeps_geometry_when_untextured() {
        // A building with no textured polygon must NOT vanish: it falls back to the highest LOD.
        // (Regression: the previous destructive probe loop left such entities with empty geometry.)
        let lods = run_textured_highest(&[(1, 0), (2, 1)], vec![None, None]);
        assert_eq!(lods, vec![2]);
    }

    #[test]
    fn textured_highest_prefers_a_textured_lod() {
        // Highest LOD (2) is untextured but a lower LOD (1) is textured -> pick the textured one.
        let lods = run_textured_highest(&[(1, 0), (2, 1)], vec![Some(0), None]);
        assert_eq!(lods, vec![1]);
        // Highest LOD (2) is textured -> pick it.
        let lods = run_textured_highest(&[(1, 0), (2, 1)], vec![None, Some(0)]);
        assert_eq!(lods, vec![2]);
    }

    #[test]
    fn test_lod_mask() {
        let mut mask = LodMask::default();
        assert_eq!(mask.lowest_lod(), None);
        assert_eq!(mask.highest_lod(), None);

        mask.add_lod(1);
        assert_eq!(mask.lowest_lod(), Some(1));
        assert_eq!(mask.highest_lod(), Some(1));
        assert!(!mask.has_lod(0));

        mask.add_lod(2);
        assert_eq!(mask.lowest_lod(), Some(1));
        assert_eq!(mask.highest_lod(), Some(2));
        assert!(!mask.has_lod(3));

        mask.add_lod(3);
        assert_eq!(mask.lowest_lod(), Some(1));
        assert_eq!(mask.highest_lod(), Some(3));
        assert!(mask.has_lod(3));

        // bitand
        let mut mask2 = LodMask::default();
        mask2.add_lod(3);
        assert!((mask & mask2).has_lod(3));
        assert!(!(mask & mask2).has_lod(1));
    }
}

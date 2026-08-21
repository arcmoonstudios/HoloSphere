/* hnsqr/src/metadata/geo.rs */
//!▫~•◦-------------------------------‣
//! # GIS GeoPolygon & 2D Spatial Filtering Engine (Front 4: Qdrant/Weaviate Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides spatial polygon bounding, ray-casting point-in-polygon intersection tests,
//! and 2D bounding boxes integrated into HNSQR's metadata filtering pipeline.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// 2D Geographic Coordinate (Latitude, Longitude) in degrees.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    pub lat: f64,
    pub lon: f64,
}

impl GeoPoint {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self { lat, lon }
    }

    /// Computes the great-circle distance between two points in kilometers (Haversine formula).
    pub fn haversine_distance_km(&self, other: &GeoPoint) -> f64 {
        let earth_radius_km = 6371.0;
        let d_lat = (other.lat - self.lat).to_radians();
        let d_lon = (other.lon - self.lon).to_radians();

        let a = (d_lat / 2.0).sin().powi(2)
            + self.lat.to_radians().cos()
                * other.lat.to_radians().cos()
                * (d_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        earth_radius_km * c
    }
}

/// Axis-aligned 2D bounding box for fast spatial pre-filtering.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox2D {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

impl BoundingBox2D {
    pub fn contains(&self, point: &GeoPoint) -> bool {
        point.lat >= self.min_lat
            && point.lat <= self.max_lat
            && point.lon >= self.min_lon
            && point.lon <= self.max_lon
    }
}

/// Arbitrary 2D GIS Polygon supporting interior/exterior boundary tests.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeoPolygon {
    pub exterior_ring: Vec<GeoPoint>,
    pub bounding_box: BoundingBox2D,
}

impl GeoPolygon {
    /// Constructs a GeoPolygon and automatically computes its 2D bounding box.
    pub fn new(exterior_ring: Vec<GeoPoint>) -> Option<Self> {
        if exterior_ring.len() < 3 {
            return None;
        }

        let mut min_lat = f64::MAX;
        let mut max_lat = f64::MIN;
        let mut min_lon = f64::MAX;
        let mut max_lon = f64::MIN;

        for p in &exterior_ring {
            min_lat = min_lat.min(p.lat);
            max_lat = max_lat.max(p.lat);
            min_lon = min_lon.min(p.lon);
            max_lon = max_lon.max(p.lon);
        }

        let bounding_box = BoundingBox2D {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        };

        Some(Self {
            exterior_ring,
            bounding_box,
        })
    }

    /// Evaluates whether a point lies strictly inside the polygon using Jordan Curve Ray-Casting.
    pub fn contains_point(&self, point: &GeoPoint) -> bool {
        // Fast bounding box rejection
        if !self.bounding_box.contains(point) {
            return false;
        }

        let mut inside = false;
        let n = self.exterior_ring.len();
        let mut j = n - 1;

        for i in 0..n {
            let pi = &self.exterior_ring[i];
            let pj = &self.exterior_ring[j];

            if ((pi.lon > point.lon) != (pj.lon > point.lon))
                && (point.lat
                    < (pj.lat - pi.lat) * (point.lon - pi.lon) / (pj.lon - pi.lon) + pi.lat)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_polygon_ray_casting() {
        // Square polygon around Manhattan (approx)
        let ring = vec![
            GeoPoint::new(40.70, -74.02),
            GeoPoint::new(40.80, -74.02),
            GeoPoint::new(40.80, -73.92),
            GeoPoint::new(40.70, -73.92),
        ];

        let polygon = GeoPolygon::new(ring).unwrap();

        let inside_point = GeoPoint::new(40.75, -73.97);
        let outside_point = GeoPoint::new(40.85, -73.97);

        assert!(polygon.contains_point(&inside_point));
        assert!(!polygon.contains_point(&outside_point));
    }
}

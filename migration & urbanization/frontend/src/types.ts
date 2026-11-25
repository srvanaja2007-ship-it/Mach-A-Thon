export interface PresetHub {
  lat: number;
  lon: number;
  description: string;
  authority: string;
}

export interface GridCell {
  lat: number;
  lon: number;
  zone: number;
  sprawl: number;
  delta_ndbi: number;
  viirs: number;
  vv_db: number;
  road_m: number;
}

export interface Corridor {
  rank: number;
  source_lat: number;
  source_lon: number;
  target_lat: number;
  target_lon: number;
  corridor_km: number;
  served_hh: number;
  cost_cr: number;
  risk_score: number;
  coords: [number, number][];
}

export interface RoiResult {
  early_budget_cr: number;
  n_funded: number;
  total_funded_cost_cr: number;
  deferred_cost_cr: number;
  net_savings_cr: number;
  net_savings_usd_m: number;
  roi_ratio: number;
  sprawl_prevented_sqkm: number;
  influx_covered: number;
  unmitigated_influx: number;
  hh_connected: number;
  coverage_ratio: number;
}

export interface RadarStat {
  zone: number;
  label: string;
  color: string;
  count: number;
  p10: number;
  mean: number;
  p90: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface AnalysisResponse {
  lat: number;
  lon: number;
  n_cells: number;
  grid_size_m: number;
  sources: Record<string, string>;
  cv_accuracy: number;
  new_built_sqkm: number;
  n_nlg_alerts: number;
  nlg_area_sqkm: number;
  influx_2027_28: number;
  sprawl_prob_mean: number;
  sprawl_prob_max: number;
  n_zones: Record<string, number>;
  zone_labels: Record<string, string>;
  zone_short: Record<string, string>;
  zone_colors: Record<string, string>;
  roi: RoiResult;
  corridors: Corridor[];
  cells: GridCell[];
  nlg_cells: GridCell[];
  radar_stats: RadarStat[];
  feat_importances: FeatureImportance[];
}

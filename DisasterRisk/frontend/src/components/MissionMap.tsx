import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Map as MapLibreMap,
  NavigationControl,
  Popup,
  ScaleControl,
  setWorkerUrl,
  type GeoJSONSource,
  type MapLayerMouseEvent,
  type StyleSpecification,
} from 'maplibre-gl';
import { feature as topojsonFeature } from 'topojson-client';
import countriesTopo from 'world-atlas/countries-110m.json';
import {
  Crosshair,
  Eye,
  Globe2,
  Layers,
  MapPin,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import type {
  DisasterType,
  EventFullResponse,
  EventSummary,
  LayerVisibility,
  SeverityLevel,
  UserPersona,
} from '../types/claimpulse';
import {
  SEVERITY_BADGE_CLASSES,
  SEVERITY_COLORS,
  formatInr,
  formatNumber,
} from '../utils/format';

setWorkerUrl('/maplibre-gl-worker.mjs');

interface MissionMapProps {
  data: EventFullResponse;
  events: EventSummary[];
  persona: UserPersona;
  layers: LayerVisibility;
  timelineIndex: number;
  focusZoneId: string | null;
  selectedDisasterType?: DisasterType;
  onInspectZone?: (zoneId: string) => void;
  onSelectPreset?: (presetId: string) => void;
}

interface InspectedFeature {
  title: string;
  subtitle: string;
  assetType: string;
  severity: SeverityLevel;
  properties: Record<string, unknown>;
}

type BasemapMode = 'dark' | 'street' | 'satellite';
type MapStage = 'after' | 'split' | 'before';

const POST_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    'esri-dark': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: '© Esri, HERE, Garmin, OpenStreetMap',
    },
    'esri-dark-ref': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
    },
    'esri-street': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: '© Esri, OpenStreetMap contributors',
    },
    'esri-satellite': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: '© Esri, Maxar, Earthstar Geographics',
    },
  },
  layers: [
    {
      id: 'ocean-bg',
      type: 'background',
      paint: { 'background-color': '#070B14' },
    },
    {
      id: 'esri-dark-layer',
      type: 'raster',
      source: 'esri-dark',
      minzoom: 0,
      maxzoom: 19,
      paint: { 'raster-opacity': 0.92 },
    },
    {
      id: 'esri-dark-ref-layer',
      type: 'raster',
      source: 'esri-dark-ref',
      minzoom: 0,
      maxzoom: 19,
      paint: { 'raster-opacity': 0.85 },
    },
    {
      id: 'esri-street-layer',
      type: 'raster',
      source: 'esri-street',
      layout: { visibility: 'none' },
      minzoom: 0,
      maxzoom: 19,
    },
    {
      id: 'esri-satellite-layer',
      type: 'raster',
      source: 'esri-satellite',
      layout: { visibility: 'none' },
      paint: { 'raster-opacity': 0.9 },
    },
  ],
};

const PRE_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    'esri-satellite-pre': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
      attribution: '© Esri Pre-Event Baseline',
    },
    'esri-ref-pre': {
      type: 'raster',
      tiles: [
        'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      ],
      tileSize: 256,
    },
  },
  layers: [
    {
      id: 'pre-bg',
      type: 'background',
      paint: { 'background-color': '#070B14' },
    },
    {
      id: 'pre-sat-layer',
      type: 'raster',
      source: 'esri-satellite-pre',
      minzoom: 0,
      maxzoom: 19,
    },
    {
      id: 'pre-ref-layer',
      type: 'raster',
      source: 'esri-ref-pre',
      minzoom: 0,
      maxzoom: 19,
      paint: { 'raster-opacity': 0.85 },
    },
  ],
};

const PERSONA_METRIC_LABEL: Record<UserPersona, string> = {
  insurer: 'Insurer Claim Est.',
  government: 'Gov. Relief Budget',
  humanitarian: 'Humanitarian Aid',
  lender: 'Collateral VaR',
  business: 'Biz Interruption',
};

export const MissionMap: React.FC<MissionMapProps> = ({
  data,
  events,
  persona,
  layers,
  timelineIndex,
  focusZoneId,
  selectedDisasterType,
  onInspectZone,
  onSelectPreset,
}) => {
  const postContainerRef = useRef<HTMLDivElement | null>(null);
  const preContainerRef = useRef<HTMLDivElement | null>(null);
  const postMapRef = useRef<MapLibreMap | null>(null);
  const preMapRef = useRef<MapLibreMap | null>(null);
  const popupRef = useRef<Popup | null>(null);

  const latestDataRef = useRef(data);
  const latestPersonaRef = useRef(persona);
  latestDataRef.current = data;
  latestPersonaRef.current = persona;

  const [mapReady, setMapReady] = useState(false);
  const [preMapReady, setPreMapReady] = useState(false);
  const [inspected, setInspected] = useState<InspectedFeature | null>(null);
  const [mapStage, setMapStage] = useState<MapStage>('after');
  const [sliderPos, setSliderPos] = useState(50);
  const [basemap, setBasemap] = useState<BasemapMode>('dark');
  const [isWorldView, setIsWorldView] = useState(false);

  const worldCountriesGeoJson = useMemo(() => {
    const topo = countriesTopo as unknown as {
      type: 'Topology';
      objects: { countries: Parameters<typeof topojsonFeature>[1] };
    };
    return topojsonFeature(topo as never, topo.objects.countries);
  }, []);

  const globalEventsGeoJson = useMemo(() => {
    const activeDisaster = selectedDisasterType || data.event.disaster_type;
    // Show only places that satisfy the selected disaster criteria
    const filteredEvents = events.filter(
      (e) =>
        e.preset_id &&
        e.intensity_multiplier === 1 &&
        e.disaster_type === activeDisaster
    );
    return {
      type: 'FeatureCollection' as const,
      features: filteredEvents.map((ev) => ({
        type: 'Feature' as const,
        id: ev.id,
        geometry: { type: 'Point' as const, coordinates: ev.center },
        properties: {
          preset_id: ev.preset_id || ev.id,
          title: ev.title,
          disaster_type: ev.disaster_type,
          district: ev.district,
          country: ev.country,
        },
      })),
    };
  }, [events, selectedDisasterType, data.event.disaster_type]);

  const latestGlobalRef = useRef(globalEventsGeoJson);
  latestGlobalRef.current = globalEventsGeoJson;

  // Initialize Post-Disaster (Primary) MapLibre instance
  useEffect(() => {
    if (!postContainerRef.current || postMapRef.current) return;

    const map = new MapLibreMap({
      container: postContainerRef.current,
      style: POST_STYLE,
      center: data.event.center,
      zoom: data.event.zoom,
      attributionControl: false,
      renderWorldCopies: true,
    });

    map.addControl(new NavigationControl({ showCompass: true }), 'top-right');
    map.addControl(new ScaleControl({ unit: 'metric' }), 'bottom-right');

    const popup = new Popup({
      closeButton: false,
      closeOnClick: false,
      maxWidth: '300px',
    });
    popupRef.current = popup;

    map.on('move', () => {
      if (preMapRef.current) {
        preMapRef.current.jumpTo({
          center: map.getCenter(),
          zoom: map.getZoom(),
          bearing: map.getBearing(),
          pitch: map.getPitch(),
        });
      }
    });

    map.on('style.load', () => {
      map.addSource('world-countries', {
        type: 'geojson',
        data: worldCountriesGeoJson as unknown as Parameters<GeoJSONSource['setData']>[0],
      });

      map.addLayer(
        {
          id: 'world-countries-fill',
          type: 'fill',
          source: 'world-countries',
          paint: {
            'fill-color': '#0E1524',
            'fill-opacity': 0.85,
          },
        },
        'esri-dark-layer'
      );

      map.addLayer({
        id: 'world-countries-border',
        type: 'line',
        source: 'world-countries',
        paint: {
          'line-color': '#22D3EE',
          'line-width': 0.7,
          'line-opacity': 0.35,
        },
      });

      map.addSource('global-events', {
        type: 'geojson',
        data: latestGlobalRef.current as unknown as Parameters<GeoJSONSource['setData']>[0],
      });

      map.addSource('hazard-zones', {
        type: 'geojson',
        data: latestDataRef.current.damage.geojson as unknown as Parameters<GeoJSONSource['setData']>[0],
      });

      map.addSource('exposure-assets', {
        type: 'geojson',
        data: latestDataRef.current.exposure.geojson as unknown as Parameters<GeoJSONSource['setData']>[0],
      });

      // Economic / Stakeholder Loss Heatmap
      map.addLayer({
        id: 'loss-heatmap-layer',
        type: 'heatmap',
        source: 'exposure-assets',
        filter: [
          'in',
          ['get', 'asset_type'],
          ['literal', ['population_cell', 'policy_zone', 'building']],
        ],
        paint: {
          'heatmap-weight': ['coalesce', ['get', 'normalized_intensity'], 0.5],
          'heatmap-intensity': 1.3,
          'heatmap-radius': 42,
          'heatmap-opacity': 0.7,
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0, 'rgba(0,0,0,0)',
            0.25, 'rgba(34,211,238,0.45)',
            0.5, 'rgba(250,204,21,0.68)',
            0.75, 'rgba(249,115,22,0.82)',
            1, 'rgba(185,28,28,0.92)',
          ],
        },
      });

      // Translucent Damage Polygons with crisp borders + hover highlight filter
      map.addLayer({
        id: 'hazard-fill-layer',
        type: 'fill',
        source: 'hazard-zones',
        paint: {
          'fill-color': [
            'match',
            ['get', 'severity'],
            'safe', SEVERITY_COLORS.safe,
            'low', SEVERITY_COLORS.low,
            'medium', SEVERITY_COLORS.medium,
            'high', SEVERITY_COLORS.high,
            'extreme', SEVERITY_COLORS.extreme,
            '#22D3EE',
          ],
          'fill-opacity': 0.42,
        },
      });

      map.addLayer({
        id: 'hazard-hover-fill',
        type: 'fill',
        source: 'hazard-zones',
        filter: ['==', ['get', 'zone_id'], ''],
        paint: {
          'fill-color': '#22D3EE',
          'fill-opacity': 0.22,
        },
      });

      map.addLayer({
        id: 'hazard-outline-layer',
        type: 'line',
        source: 'hazard-zones',
        paint: {
          'line-color': [
            'match',
            ['get', 'severity'],
            'safe', '#4ADE80',
            'low', '#FDE047',
            'medium', '#FB923C',
            'high', '#F87171',
            'extreme', '#FCA5A5',
            '#22D3EE',
          ],
          'line-width': 2,
          'line-opacity': 0.95,
        },
      });

      map.addLayer({
        id: 'hazard-hover-outline',
        type: 'line',
        source: 'hazard-zones',
        filter: ['==', ['get', 'zone_id'], ''],
        paint: {
          'line-color': '#F4F7FB',
          'line-width': 3.5,
        },
      });

      map.addLayer({
        id: 'farmland-layer',
        type: 'fill',
        source: 'exposure-assets',
        filter: ['==', ['get', 'asset_type'], 'farmland'],
        paint: {
          'fill-color': '#22C55E',
          'fill-opacity': 0.34,
          'fill-outline-color': '#4ADE80',
        },
      });

      map.addLayer({
        id: 'buildings-layer',
        type: 'fill',
        source: 'exposure-assets',
        filter: ['==', ['get', 'asset_type'], 'building'],
        paint: {
          'fill-color': '#2563EB',
          'fill-opacity': 0.65,
          'fill-outline-color': '#93C5FD',
        },
      });

      map.addLayer({
        id: 'roads-layer',
        type: 'line',
        source: 'exposure-assets',
        filter: ['==', ['get', 'asset_type'], 'road'],
        paint: {
          'line-color': '#F59E0B',
          'line-width': 3,
          'line-dasharray': [2, 1],
          'line-opacity': 0.92,
        },
      });

      map.addLayer({
        id: 'population-layer',
        type: 'circle',
        source: 'exposure-assets',
        filter: ['==', ['get', 'asset_type'], 'population_cell'],
        paint: {
          'circle-radius': [
            'interpolate',
            ['linear'],
            ['coalesce', ['get', 'population_exposed'], 5000],
            2000, 9,
            25000, 20,
            80000, 32,
          ],
          'circle-color': '#8B5CF6',
          'circle-opacity': 0.34,
          'circle-stroke-color': '#C4B5FD',
          'circle-stroke-width': 1.2,
        },
      });

      map.addLayer({
        id: 'portfolio-layer',
        type: 'circle',
        source: 'exposure-assets',
        filter: ['==', ['get', 'asset_type'], 'policy_zone'],
        paint: {
          'circle-radius': 7,
          'circle-color': '#FACC15',
          'circle-stroke-color': '#070B14',
          'circle-stroke-width': 2,
        },
      });

      map.addLayer({
        id: 'critical-infra-layer',
        type: 'circle',
        source: 'exposure-assets',
        filter: ['==', ['get', 'asset_type'], 'critical_infrastructure'],
        paint: {
          'circle-radius': 8,
          'circle-color': '#22D3EE',
          'circle-stroke-color': '#F4F7FB',
          'circle-stroke-width': 2,
        },
      });

      map.addLayer({
        id: 'global-events-pulse',
        type: 'circle',
        source: 'global-events',
        maxzoom: 7.5,
        paint: {
          'circle-radius': 15,
          'circle-color': [
            'match',
            ['get', 'disaster_type'],
            'wildfire', '#F97316',
            'cyclone', '#8B5CF6',
            '#22D3EE',
          ],
          'circle-opacity': 0.25,
        },
      });

      map.addLayer({
        id: 'global-events-pin',
        type: 'circle',
        source: 'global-events',
        maxzoom: 7.5,
        paint: {
          'circle-radius': 7.5,
          'circle-color': [
            'match',
            ['get', 'disaster_type'],
            'wildfire', '#F97316',
            'cyclone', '#8B5CF6',
            '#2563EB',
          ],
          'circle-stroke-color': '#F4F7FB',
          'circle-stroke-width': 2,
        },
      });

      map.on('mousemove', 'global-events-pin', (e: MapLayerMouseEvent) => {
        map.getCanvas().style.cursor = 'pointer';
        const feat = e.features?.[0];
        if (!feat?.properties) return;
        const p = feat.properties;
        popup
          .setLngLat(e.lngLat)
          .setHTML(
            `<div style="font-size:12px">
              <div style="font-weight:600;color:#F4F7FB">${p.title}</div>
              <div style="color:#9FB0C7;font-size:11px">${p.district}, ${p.country}</div>
              <div style="color:#22D3EE;font-size:10px;margin-top:3px">Click pin to focus event</div>
            </div>`
          )
          .addTo(map);
      });

      map.on('mouseleave', 'global-events-pin', () => {
        map.getCanvas().style.cursor = '';
        popup.remove();
      });

      map.on('click', 'global-events-pin', (e: MapLayerMouseEvent) => {
        const pid = e.features?.[0]?.properties?.preset_id;
        if (pid && onSelectPreset) {
          setIsWorldView(false);
          onSelectPreset(String(pid));
        }
      });

      const interactiveLayers = [
        'critical-infra-layer',
        'portfolio-layer',
        'buildings-layer',
        'farmland-layer',
        'roads-layer',
        'hazard-fill-layer',
      ];

      interactiveLayers.forEach((layerId) => {
        map.on('mousemove', layerId, (e: MapLayerMouseEvent) => {
          map.getCanvas().style.cursor = 'pointer';
          const feat = e.features?.[0];
          if (!feat?.properties) return;
          const p = feat.properties;
          const zid = String(p.zone_id || '');
          if (zid) {
            map.setFilter('hazard-hover-fill', ['==', ['get', 'zone_id'], zid]);
            map.setFilter('hazard-hover-outline', ['==', ['get', 'zone_id'], zid]);
          }

          const sev = (p.severity as SeverityLevel) || 'medium';
          const sevColor = SEVERITY_COLORS[sev] || '#22D3EE';
          const activePersona = latestPersonaRef.current;
          const personaAmt = Number(p[`${activePersona}_inr`] || p.exposure_value_inr || 0);
          const personaLabel = PERSONA_METRIC_LABEL[activePersona];
          const preState = p.pre_state ? String(p.pre_state) : '';
          const postState = p.post_state ? String(p.post_state) : '';

          popup
            .setLngLat(e.lngLat)
            .setHTML(
              `<div style="font-size:12px;line-height:1.4">
                <div style="font-weight:600;color:#F4F7FB;margin-bottom:3px">${p.name || 'Zone'}</div>
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">
                  <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${sevColor}"></span>
                  <span style="text-transform:uppercase;font-size:10px;letter-spacing:0.05em;color:#F4F7FB;font-weight:600">${sev} RISK</span>
                  ${p.hazard_intensity !== undefined ? `<span style="color:#22D3EE;font-family:'IBM Plex Mono',monospace;font-size:11px">• ${p.hazard_intensity}</span>` : ''}
                </div>
                ${preState ? `<div style="font-size:10px;color:#9FB0C7"><b style="color:#22C55E">Before:</b> ${preState}</div>` : ''}
                ${postState ? `<div style="font-size:10px;color:#9FB0C7;margin-bottom:4px"><b style="color:#EF4444">After:</b> ${postState}</div>` : ''}
                ${personaAmt > 0 ? `<div style="font-size:11px;color:#F4F7FB;border-top:1px solid #1E2A40;padding-top:4px;margin-top:4px">${personaLabel}: <b style="font-family:'IBM Plex Mono',monospace;color:#22D3EE">${formatInr(personaAmt)}</b></div>` : ''}
              </div>`
            )
            .addTo(map);
        });

        map.on('mouseleave', layerId, () => {
          map.getCanvas().style.cursor = '';
          map.setFilter('hazard-hover-fill', ['==', ['get', 'zone_id'], '']);
          map.setFilter('hazard-hover-outline', ['==', ['get', 'zone_id'], '']);
          popup.remove();
        });

        map.on('click', layerId, (e: MapLayerMouseEvent) => {
          const feat = e.features?.[0];
          if (!feat?.properties) return;
          const p = feat.properties;
          setInspected({
            title: String(p.name || 'Selected Feature'),
            subtitle: String(p.zone_name || p.district || p.critical_category || 'Hazard Zone'),
            assetType: String(p.asset_type || 'hazard_zone'),
            severity: (p.severity as SeverityLevel) || 'medium',
            properties: p,
          });
          if (p.zone_id && onInspectZone) {
            onInspectZone(String(p.zone_id));
          }
        });
      });

      setMapReady(true);
    });

    postMapRef.current = map;

    return () => {
      map.remove();
      postMapRef.current = null;
    };
  }, [worldCountriesGeoJson]);

  // Initialize Synchronized Pre-Disaster (Before) MapLibre instance
  useEffect(() => {
    if (!preContainerRef.current || preMapRef.current) return;

    const preMap = new MapLibreMap({
      container: preContainerRef.current,
      style: PRE_STYLE,
      center: data.event.center,
      zoom: data.event.zoom,
      attributionControl: false,
      interactive: true,
      renderWorldCopies: true,
    });

    // Sync user panning/zooming on the left side back to primary map in Before/Split mode
    preMap.on('move', () => {
      const post = postMapRef.current;
      if (post) {
        const pc = preMap.getCenter();
        const oc = post.getCenter();
        if (
          Math.abs(pc.lng - oc.lng) > 1e-5 ||
          Math.abs(pc.lat - oc.lat) > 1e-5 ||
          Math.abs(preMap.getZoom() - post.getZoom()) > 1e-4
        ) {
          post.jumpTo({
            center: pc,
            zoom: preMap.getZoom(),
            bearing: preMap.getBearing(),
            pitch: preMap.getPitch(),
          });
        }
      }
    });

    preMap.on('style.load', () => {
      preMap.addSource('pre-zones', {
        type: 'geojson',
        data: latestDataRef.current.damage.geojson as unknown as Parameters<GeoJSONSource['setData']>[0],
      });
      preMap.addSource('pre-assets', {
        type: 'geojson',
        data: latestDataRef.current.exposure.geojson as unknown as Parameters<GeoJSONSource['setData']>[0],
      });

      // Pre-disaster intact green zone boundaries (no flood/burn fill!)
      preMap.addLayer({
        id: 'pre-zones-outline',
        type: 'line',
        source: 'pre-zones',
        paint: {
          'line-color': '#22C55E',
          'line-width': 1.4,
          'line-dasharray': [3, 2],
          'line-opacity': 0.75,
        },
      });

      preMap.addLayer({
        id: 'pre-farmland',
        type: 'fill',
        source: 'pre-assets',
        filter: ['==', ['get', 'asset_type'], 'farmland'],
        paint: {
          'fill-color': '#22C55E',
          'fill-opacity': 0.28,
          'fill-outline-color': '#4ADE80',
        },
      });

      preMap.addLayer({
        id: 'pre-roads',
        type: 'line',
        source: 'pre-assets',
        filter: ['==', ['get', 'asset_type'], 'road'],
        paint: {
          'line-color': '#E2E8F0',
          'line-width': 2.5,
          'line-opacity': 0.88,
        },
      });

      preMap.addLayer({
        id: 'pre-facilities',
        type: 'circle',
        source: 'pre-assets',
        filter: ['==', ['get', 'asset_type'], 'critical_infrastructure'],
        paint: {
          'circle-radius': 6.5,
          'circle-color': '#22C55E',
          'circle-stroke-color': '#F4F7FB',
          'circle-stroke-width': 1.5,
        },
      });

      setPreMapReady(true);
    });

    preMapRef.current = preMap;

    return () => {
      preMap.remove();
      preMapRef.current = null;
    };
  }, []);

  // Update GeoJSON data & flyTo when event changes
  useEffect(() => {
    const map = postMapRef.current;
    if (map && mapReady) {
      (map.getSource('hazard-zones') as GeoJSONSource | undefined)?.setData(
        data.damage.geojson as unknown as Parameters<GeoJSONSource['setData']>[0]
      );
      (map.getSource('exposure-assets') as GeoJSONSource | undefined)?.setData(
        data.exposure.geojson as unknown as Parameters<GeoJSONSource['setData']>[0]
      );
      (map.getSource('global-events') as GeoJSONSource | undefined)?.setData(
        globalEventsGeoJson as unknown as Parameters<GeoJSONSource['setData']>[0]
      );
      map.flyTo({ center: data.event.center, zoom: data.event.zoom, duration: 750 });
    }

    const preMap = preMapRef.current;
    if (preMap && preMapReady) {
      (preMap.getSource('pre-zones') as GeoJSONSource | undefined)?.setData(
        data.damage.geojson as unknown as Parameters<GeoJSONSource['setData']>[0]
      );
      (preMap.getSource('pre-assets') as GeoJSONSource | undefined)?.setData(
        data.exposure.geojson as unknown as Parameters<GeoJSONSource['setData']>[0]
      );
    }

    setInspected(null);
    setIsWorldView(false);
  }, [data, globalEventsGeoJson, mapReady, preMapReady]);

  // Focus zone when user clicks a Priority Zone row
  useEffect(() => {
    const map = postMapRef.current;
    if (!map || !mapReady || !focusZoneId) return;
    const feat = data.damage.geojson.features.find(
      (f) => f.properties.zone_id === focusZoneId
    );
    if (feat && Array.isArray(feat.properties.center)) {
      setIsWorldView(false);
      map.flyTo({
        center: feat.properties.center as [number, number],
        zoom: Math.max(data.event.zoom + 1.1, 11.2),
        duration: 650,
      });
      setInspected({
        title: String(feat.properties.name),
        subtitle: String(feat.properties.district),
        assetType: 'hazard_zone',
        severity: (feat.properties.severity as SeverityLevel) || 'high',
        properties: feat.properties,
      });
    }
  }, [focusZoneId, mapReady, data]);

  // Update layer visibility, basemap, and timeline progression
  useEffect(() => {
    const map = postMapRef.current;
    if (!map || !mapReady) return;

    const setVis = (id: string, visible: boolean) => {
      if (map.getLayer(id)) {
        map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
      }
    };

    const effectiveMode: BasemapMode = layers.satelliteImagery ? 'satellite' : basemap;
    setVis('esri-dark-layer', effectiveMode === 'dark');
    setVis('esri-dark-ref-layer', effectiveMode === 'dark');
    setVis('esri-street-layer', effectiveMode === 'street');
    setVis('esri-satellite-layer', effectiveMode === 'satellite');

    setVis('hazard-fill-layer', layers.hazardExtent);
    setVis('hazard-outline-layer', layers.hazardExtent);
    setVis('buildings-layer', layers.buildings);
    setVis('population-layer', layers.population);
    setVis('roads-layer', layers.roads);
    setVis('farmland-layer', layers.farmland);
    setVis('critical-infra-layer', layers.criticalInfra);
    setVis('portfolio-layer', layers.insurancePortfolio);
    setVis('loss-heatmap-layer', layers.lossHeatmap);

    const totalSteps = Math.max(1, data.damage.metrics.timeline_dates.length - 1);
    const progress = totalSteps > 0 ? Math.max(0.25, timelineIndex / totalSteps) : 1;
    if (map.getLayer('hazard-fill-layer')) {
      map.setPaintProperty('hazard-fill-layer', 'fill-opacity', 0.2 + 0.3 * progress);
    }
  }, [layers, basemap, timelineIndex, mapReady, data]);

  // Stakeholder emphasis on map layers
  useEffect(() => {
    const map = postMapRef.current;
    if (!map || !mapReady) return;
    if (map.getLayer('roads-layer')) {
      map.setPaintProperty(
        'roads-layer',
        'line-width',
        persona === 'government' || persona === 'business' ? 4.2 : 2.8
      );
    }
    if (map.getLayer('population-layer')) {
      map.setPaintProperty(
        'population-layer',
        'circle-opacity',
        persona === 'humanitarian' || persona === 'government' ? 0.55 : 0.28
      );
    }
    if (map.getLayer('portfolio-layer')) {
      map.setPaintProperty(
        'portfolio-layer',
        'circle-radius',
        persona === 'insurer' || persona === 'lender' ? 8.5 : 6
      );
    }
  }, [persona, mapReady]);

  const handleWorldView = () => {
    const map = postMapRef.current;
    if (!map) return;
    setIsWorldView(true);
    map.flyTo({ center: [-10, 22], zoom: 1.15, duration: 850 });
  };

  const handleFocusEvent = () => {
    const map = postMapRef.current;
    if (!map) return;
    setIsWorldView(false);
    map.flyTo({ center: data.event.center, zoom: data.event.zoom, duration: 750 });
  };

  const comp = data.damage.metrics.imagery_comparison;
  const activeStakeholder = data.loss.stakeholder_models?.[persona];
  const postClipStyle =
    mapStage === 'split'
      ? { clipPath: `inset(0 0 0 ${sliderPos}%)` }
      : mapStage === 'before'
      ? { clipPath: 'inset(0 0 0 100%)' }
      : { clipPath: 'inset(0 0 0 0)' };

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#070B14]" aria-label="Interactive disaster risk map">
      {/* Synchronized Pre-Disaster Map Layer (Bottom z-0) */}
      <div className="absolute inset-0 z-0 h-full w-full overflow-hidden">
        <div ref={preContainerRef} className="h-full w-full" />
      </div>

      {/* Synchronized Post-Disaster Impact Map Layer (Top z-[1], clipped cleanly in Split / Before mode) */}
      <div
        className="absolute inset-0 z-[1] h-full w-full overflow-hidden"
        style={postClipStyle}
      >
        <div ref={postContainerRef} className="h-full w-full" />
      </div>

      {/* Split Divider Line */}
      {mapStage === 'split' && (
        <div
          className="pointer-events-none absolute inset-y-0 z-10 w-0.5 bg-[#22D3EE]"
          style={{ left: `${sliderPos}%` }}
        >
          <div className="glass-float absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full px-2.5 py-1 font-mono-tabular text-[10px] font-semibold text-[#F4F7FB] whitespace-nowrap shadow-lg">
            BEFORE {sliderPos}% | AFTER {100 - sliderPos}%
          </div>
        </div>
      )}

      {/* Top-Left Floating Command Controls */}
      <div className="pointer-events-none absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2 pr-14">
        <div className="pointer-events-auto glass-float flex items-center gap-2 rounded-lg px-3 py-1.5 shadow-lg">
          <span className="text-xs font-semibold text-[#F4F7FB]">{data.event.title}</span>
          <span className="rounded bg-[#141C2E] border border-[#1E2A40] px-1.5 py-0.5 font-mono-tabular text-[10px] text-[#22D3EE]">
            {data.damage.metrics.max_intensity_label}
          </span>
        </div>

        {/* Before / Split / After Toggle */}
        <div
          className="pointer-events-auto glass-float flex items-center rounded-lg p-0.5 text-[11px]"
          role="group"
          aria-label="Map disaster timeline comparison mode"
        >
          {(
            [
              { id: 'before', label: 'Before' },
              { id: 'split', label: 'Split Compare' },
              { id: 'after', label: 'After Impact' },
            ] as const
          ).map((st) => (
            <button
              key={st.id}
              type="button"
              onClick={() => setMapStage(st.id)}
              className={`flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors duration-150 ${
                mapStage === st.id
                  ? 'bg-[#2563EB] text-[#F4F7FB]'
                  : 'text-[#9FB0C7] hover:text-[#F4F7FB]'
              }`}
            >
              {st.id === 'split' && <SlidersHorizontal className="h-3 w-3" />}
              {st.label}
            </button>
          ))}
        </div>

        {/* World View / Focus Event */}
        <div className="pointer-events-auto glass-float flex items-center rounded-lg p-0.5 text-[11px]">
          <button
            type="button"
            onClick={handleWorldView}
            className={`flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors duration-150 ${
              isWorldView ? 'bg-[#2563EB] text-[#F4F7FB]' : 'text-[#9FB0C7] hover:text-[#F4F7FB]'
            }`}
          >
            <Globe2 className="h-3.5 w-3.5" /> World
          </button>
          <button
            type="button"
            onClick={handleFocusEvent}
            className={`flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors duration-150 ${
              !isWorldView ? 'bg-[#141C2E] text-[#F4F7FB] border border-[#1E2A40]' : 'text-[#9FB0C7] hover:text-[#F4F7FB]'
            }`}
          >
            <Crosshair className="h-3.5 w-3.5 text-[#22D3EE]" /> Focus
          </button>
        </div>

        {/* Basemap Style */}
        <div className="pointer-events-auto glass-float flex items-center rounded-lg p-0.5 text-[11px]">
          {(
            [
              { id: 'dark', label: 'Dark' },
              { id: 'street', label: 'Street' },
              { id: 'satellite', label: 'Satellite' },
            ] as const
          ).map((b) => (
            <button
              key={b.id}
              type="button"
              onClick={() => setBasemap(b.id)}
              className={`rounded-md px-2 py-1 font-medium transition-colors duration-150 ${
                (layers.satelliteImagery ? 'satellite' : basemap) === b.id
                  ? 'bg-[#141C2E] text-[#22D3EE] border border-[#1E2A40]'
                  : 'text-[#9FB0C7] hover:text-[#F4F7FB]'
              }`}
            >
              {b.label}
            </button>
          ))}
        </div>
      </div>

      {/* Before / After State Callout Badges */}
      {mapStage !== 'after' && (
        <div className="pointer-events-none absolute top-[88px] left-3 z-20 rounded-lg border border-[#22C55E]/50 bg-[#0E1524]/92 px-3 py-1.5 text-[11px] backdrop-blur-md">
          <div className="font-semibold text-[#22C55E]">
            BEFORE DISASTER ({comp.pre_event_date})
          </div>
          <div className="text-[#9FB0C7] text-[10px]">
            Intact satellite baseline • Passable roads • Dry floodplain
          </div>
        </div>
      )}

      {mapStage === 'split' && (
        <div className="pointer-events-none absolute top-[88px] right-14 z-20 rounded-lg border border-[#EF4444]/50 bg-[#0E1524]/92 px-3 py-1.5 text-[11px] text-right backdrop-blur-md">
          <div className="font-semibold text-[#EF4444]">
            AFTER DISASTER ({comp.post_event_date})
          </div>
          <div className="text-[#9FB0C7] text-[10px]">
            {data.damage.metrics.affected_area_km2} km² impacted •{' '}
            {activeStakeholder
              ? `${activeStakeholder.primary_metric_title}: ${formatInr(
                  activeStakeholder.primary_amount_inr
                )}`
              : ''}
          </div>
        </div>
      )}

      {/* Split Slider Control Bar */}
      {mapStage === 'split' && (
        <div className="pointer-events-auto glass-float absolute bottom-12 left-1/2 z-20 flex w-80 -translate-x-1/2 items-center gap-3 rounded-full px-4 py-2 shadow-xl">
          <span className="font-mono-tabular text-[10px] font-semibold text-[#22C55E]">BEFORE</span>
          <input
            type="range"
            min={10}
            max={90}
            value={sliderPos}
            onChange={(e) => setSliderPos(Number(e.target.value))}
            className="w-full accent-[#2563EB] cursor-ew-resize"
            aria-label="Slide to compare before and after disaster satellite map"
          />
          <span className="font-mono-tabular text-[10px] font-semibold text-[#EF4444]">AFTER</span>
        </div>
      )}

      {/* Active Stakeholder Map Impact Pill + Severity Legend */}
      <div className="pointer-events-auto glass-float absolute bottom-3 left-3 z-20 flex flex-wrap items-center gap-3 rounded-lg px-3 py-1.5 text-[11px] shadow-lg">
        <span className="flex items-center gap-1 text-[#9FB0C7] font-medium">
          <Layers className="h-3.5 w-3.5 text-[#22D3EE]" /> Severity:
        </span>
        {(['safe', 'low', 'medium', 'high', 'extreme'] as SeverityLevel[]).map((sev) => (
          <span key={sev} className="flex items-center gap-1 capitalize text-[#F4F7FB]">
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: SEVERITY_COLORS[sev] }}
            />
            {sev}
          </span>
        ))}
        {activeStakeholder && (
          <span className="border-l border-[#1E2A40] pl-3 font-mono-tabular text-[11px] text-[#22D3EE]">
            {PERSONA_METRIC_LABEL[persona]}: <b>{formatInr(activeStakeholder.primary_amount_inr)}</b>
          </span>
        )}
      </div>

      {/* Click-to-Inspect Evidence Card */}
      {inspected && (
        <div className="glass-float absolute bottom-12 right-3 z-30 w-80 rounded-xl p-3.5 shadow-2xl">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-1.5">
                <MapPin className="h-3.5 w-3.5 text-[#22D3EE] shrink-0" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#22D3EE]">
                  {inspected.assetType.replace('_', ' ')}
                </span>
              </div>
              <h4 className="mt-0.5 text-sm font-semibold text-[#F4F7FB] leading-snug">
                {inspected.title}
              </h4>
              <p className="text-[11px] text-[#9FB0C7]">{inspected.subtitle}</p>
            </div>
            <button
              type="button"
              onClick={() => setInspected(null)}
              className="rounded p-1 text-[#9FB0C7] hover:bg-[#141C2E] hover:text-[#F4F7FB]"
              aria-label="Close evidence panel"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="mt-2.5 flex items-center justify-between border-y border-[#1E2A40] py-2">
            <span
              className={`rounded border px-2 py-0.5 text-[10px] font-semibold uppercase ${
                SEVERITY_BADGE_CLASSES[inspected.severity]
              }`}
            >
              {inspected.severity} risk
            </span>
            <span className="font-mono-tabular text-xs text-[#9FB0C7]">
              Conf:{' '}
              <b className="text-[#22D3EE]">
                {Math.round(Number(inspected.properties.confidence || 0.9) * 100)}%
              </b>
            </span>
          </div>

          {/* Before / After Comparison inside Inspector */}
          {inspected.properties.pre_state !== undefined && (
            <div className="mt-2.5 space-y-1 rounded-lg border border-[#1E2A40] bg-[#070B14]/80 p-2 text-[11px]">
              <div>
                <span className="font-semibold text-[#22C55E]">Before: </span>
                <span className="text-[#9FB0C7]">{String(inspected.properties.pre_state)}</span>
              </div>
              <div>
                <span className="font-semibold text-[#EF4444]">After: </span>
                <span className="text-[#F4F7FB]">{String(inspected.properties.post_state)}</span>
              </div>
            </div>
          )}

          <div className="mt-2.5 space-y-1.5 text-xs">
            {inspected.properties[`${persona}_inr`] !== undefined && (
              <div className="flex justify-between">
                <span className="text-[#9FB0C7]">{PERSONA_METRIC_LABEL[persona]}</span>
                <span className="font-mono-tabular font-semibold text-[#22D3EE]">
                  {formatInr(Number(inspected.properties[`${persona}_inr`]))}
                </span>
              </div>
            )}
            {inspected.properties.exposure_value_inr !== undefined && (
              <div className="flex justify-between">
                <span className="text-[#9FB0C7]">Asset Exposure</span>
                <span className="font-mono-tabular text-[#F4F7FB]">
                  {formatInr(Number(inspected.properties.exposure_value_inr))}
                </span>
              </div>
            )}
            {inspected.properties.area_km2 !== undefined && (
              <div className="flex justify-between">
                <span className="text-[#9FB0C7]">Zone Area</span>
                <span className="font-mono-tabular text-[#F4F7FB]">
                  {String(inspected.properties.area_km2)} km²
                </span>
              </div>
            )}
            {inspected.properties.building_count !== undefined && (
              <div className="flex justify-between">
                <span className="text-[#9FB0C7]">OSM Buildings</span>
                <span className="font-mono-tabular text-[#F4F7FB]">
                  {formatNumber(Number(inspected.properties.building_count))}
                </span>
              </div>
            )}
            {inspected.properties.area_ha !== undefined && (
              <div className="flex justify-between">
                <span className="text-[#9FB0C7]">Cropland</span>
                <span className="font-mono-tabular text-[#F4F7FB]">
                  {formatNumber(Number(inspected.properties.area_ha))} ha
                </span>
              </div>
            )}
            {inspected.properties.length_km !== undefined && (
              <div className="flex justify-between">
                <span className="text-[#9FB0C7]">Road Corridor</span>
                <span className="font-mono-tabular text-[#F4F7FB]">
                  {String(inspected.properties.length_km)} km
                </span>
              </div>
            )}
          </div>

          <div className="mt-2.5 flex items-center justify-between text-[10px] text-[#9FB0C7] border-t border-[#1E2A40] pt-2">
            <span className="flex items-center gap-1">
              <Eye className="h-3 w-3 text-[#22D3EE]" /> Sentinel-1 + OSM + DEM
            </span>
            <span className="font-mono-tabular">{data.event.data_freshness.slice(0, 10)}</span>
          </div>
        </div>
      )}
    </div>
  );
};

import geopandas as gpd
import shapely

baku = gpd.read_file("../../data/baku.geojson")
districts_gdf = gpd.read_file("../../data/baku_districts.geojson")
output_path = "../../data/baku_grid.geojson"
CELL_SIZE = 1000 #in meters; need 1 km x 1 km grid cells
DISTRICT_NAME_COL = "wikipedia"

if baku.crs is None:
    baku = baku.set_crs("EPSG:4326")

if districts_gdf.crs is None:
    districts_gdf = districts_gdf.set_crs("EPSG:4326")

baku_m = baku.to_crs("EPSG:32639")
districts_m = districts_gdf.to_crs("EPSG:32639")

if DISTRICT_NAME_COL not in districts_m.columns:
    raise ValueError(f"{DISTRICT_NAME_COL} not found. Available columns: {districts_m.columns}")

def find_intersection(cell, district):
    intersection = cell.intersection(district)
    if not intersection.is_empty:
        return intersection.area
    return 0

def find_district_by_largest_overlap(cell, districts_gdf):
    max_intersection = 0
    ans = "Nərimanov rayonu"  # Default district if no intersection is found, as Narimanov isn't included in the districts_gdf
    for _, district_row in districts_gdf.iterrows():
        district = district_row.geometry
        cur = find_intersection(cell, district)
        if cur > max_intersection:
            max_intersection = cur
            district_name = district_row[DISTRICT_NAME_COL]
            if isinstance(district_name, str) and ":" in district_name:
                ans = district_name.split(":", 1)[1]
            else:
                ans = district_name
            ans = district_row[DISTRICT_NAME_COL].split(":")[1]
    return ans

def create_grid(city_gdf, districts_gdf, cell_size):
    city_polygon = city_gdf.geometry.union_all() 
    minx, miny, maxx, maxy = city_polygon.bounds
    grid_cells = []
    cell_counter = 0
    x = minx

    while x < maxx:
        y = miny
        while y < maxy:
            geom = shapely.geometry.box(x, y, x + cell_size, y + cell_size)
            if not city_polygon.intersects(geom):
                y += cell_size
                continue
            district_name = find_district_by_largest_overlap(geom, districts_gdf)
            grid_cells.append({
                'geometry': geom,
                'cell_id': cell_counter, 
                'district': district_name,
                'images_collected': 0, 
            })
            cell_counter += 1
            y += cell_size
        x += cell_size
            
    grid_gdf = gpd.GeoDataFrame(grid_cells, crs=city_gdf.crs)
    centroids_wgs84 = gpd.GeoSeries(grid_gdf.geometry.centroid,crs=city_gdf.crs).to_crs("EPSG:4326")
    grid_gdf["centroid_lon"] = centroids_wgs84.x
    grid_gdf["centroid_lat"] = centroids_wgs84.y

    grid_gdf = grid_gdf.to_crs("EPSG:4326")
    bounds = grid_gdf.bounds
    grid_gdf["west"] = bounds["minx"]
    grid_gdf["south"] = bounds["miny"]
    grid_gdf["east"] = bounds["maxx"]
    grid_gdf["north"] = bounds["maxy"]
    return grid_gdf

grid_gdf = create_grid(baku_m, districts_m, CELL_SIZE)
grid_gdf.to_file(output_path, driver="GeoJSON")
print(f"Created {len(grid_gdf)} grid cells")
import geopandas as gpd
import shapely
import mapillary.interface as mly
import pandas as pd
import json
import time

MLY_ACCESS_TOKEN = ""
grid_path = "../../data/baku_grid.geojson"
grid_gdf= gpd.read_file(grid_path)
output_path = "../../data/mapillary_images_metadata.csv"
SOURCE_NAME = "Mapillary"  
WROTE_HEADER = False 

def process_grid(grid_gdf, output_path, wrote_header):
    mly.set_access_token(MLY_ACCESS_TOKEN)
    mapillary_images = []
    errors_ids = []
    for idx, row in grid_gdf.iterrows():
        west, south, east, north = row['west'], row['south'], row['east'], row['north']
        try:
            cell_images_str = mly.images_in_bbox(bbox = {"west":west, "south":south, "east":east, "north":north}, image_type="flat")
            cell_images_dict = json.loads(cell_images_str)
            cell_images = gpd.GeoDataFrame.from_features(cell_images_dict['features'])
            if cell_images.empty:
                continue
            for _, img_row in cell_images.iterrows():
                mapillary_images.append({
                    "image_id": img_row['id'],
                    "cell_id": row['cell_id'],
                    "sequence_id": img_row['sequence_id'], 
                    "district": row['district'],
                    "lat": img_row['geometry'].y,
                    "lon": img_row['geometry'].x,
                    "source": SOURCE_NAME, 
                    "captured_at": img_row['captured_at'],
                    'quality_score': img_row['quality_score']
                })

            if len(mapillary_images) > 200:
                print(f"Processed {idx + 1} grid cells, collected {len(mapillary_images)} images so far...")
                if mapillary_images:
                    images_df = pd.DataFrame(mapillary_images)
                    images_df.to_csv(
                        output_path,
                        index=False,
                        mode="a",
                        header=not wrote_header
                    )
                    wrote_header= True
                    mapillary_images= []  # Clear the list to free memory
        except Exception as e:
            print(f"Error processing grid cell {row['cell_id']}: {e}")
            errors_ids.append(row['cell_id'])
            time.sleep(1) 
            continue
    if mapillary_images:
        images_df = pd.DataFrame(mapillary_images)
        images_df.to_csv(
            output_path,
            index=False,
            mode="a",
            header=not wrote_header
        )
        wrote_header = True
    print("Error occurred for the following grid cells:", errors_ids)
    return errors_ids, wrote_header
    

error_ids, WROTE_HEADER =  process_grid(grid_gdf, output_path, WROTE_HEADER)
print(f"Metadata for images collected and saved to {output_path}.")

def process_errors(grid_gdf, output_path, error_ids, wrote_header):
    mly.set_access_token(MLY_ACCESS_TOKEN)
    mapillary_images = []
    error_ids = []
    for cell_id in error_ids:
        row = grid_gdf[grid_gdf['cell_id'] == cell_id].iloc[0]
        west, south, east, north = row['west'], row['south'], row['east'], row['north']
        try:
            cell_images_str = mly.images_in_bbox(bbox = {"west":west, "south":south, "east":east, "north":north}, image_type="flat")
            cell_images_dict = json.loads(cell_images_str)
            cell_images = gpd.GeoDataFrame.from_features(cell_images_dict['features'])
            if cell_images.empty:
                continue
            for _, img_row in cell_images.iterrows():
                mapillary_images.append({
                    "image_id": img_row['id'],
                    "cell_id": row['cell_id'],
                    "sequence_id": img_row['sequence_id'], 
                    "district": row['district'],
                    "lat": img_row['geometry'].y,
                    "lon": img_row['geometry'].x,
                    "source": SOURCE_NAME,
                    "captured_at": img_row['captured_at'],
                    'quality_score': img_row['quality_score']
                })
        except Exception as e:
            print(f"Error processing grid cell {row['cell_id']}: {e}")
            error_ids.append(row['cell_id'])
            time.sleep(1) 
            continue
    if mapillary_images:
        images_df = pd.DataFrame(mapillary_images)
        images_df.to_csv(
            output_path,
            index=False,
            mode="a",
            header=not wrote_header
        )
    # print("Finished processing error cells.")
    return error_ids

error_ids = process_errors(grid_gdf, output_path, error_ids, WROTE_HEADER) #can loop until error_ids is empty
print(f"Finished processing all error cells. Remaining errors: {error_ids}")

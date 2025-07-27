import datetime
import urllib.parse
import re

def to_geojson(img_url: str, boxes):
    m = re.search(r"lat([-\d\.]+)_lon([-\d\.]+)", urllib.parse.unquote(img_url))
    if not m:
        raise ValueError("image_url must encode lat/lon for geo mode")
    lat0, lon0 = map(float, m.groups())     # centre of image
    # very naïve mapping: treat  image as 640×640 m for demo purposes
    features=[]
    for x1,y1,x2,y2,conf in boxes:
        relx, rely = ((x1+x2)/2-320)/320, ((y1+y2)/2-320)/320
        features.append({
            "type":"Feature",
            "properties":{"conf":float(conf)},
            "geometry":{"type":"Point",
                        "coordinates":[lon0+relx*0.005, lat0-rely*0.005]}
        })
    return {"type":"FeatureCollection", "features":features,
            "timestamp": datetime.datetime.utcnow().isoformat()+"Z"}

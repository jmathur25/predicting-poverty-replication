'''
Handful of utility functions used throughout the repo
'''

import math
import pandas as pd

def merge_on_lat_lon(df1, df2, keys=['cluster_lat', 'cluster_lon'], how='inner'):
    """
    Allows two dataframes to be merged on lat/lon
    Necessary because pandas has trouble merging on floats (understandably so)
    """
    df1 = df1.copy()
    df2 = df2.copy()
    
    # must use ints for merging, as floats induce errors
    df1['merge_lat'] = (10000 * df1[keys[0]]).astype(int)
    df1['merge_lon'] = (10000 * df1[keys[1]]).astype(int)
    
    df2['merge_lat'] = (10000 * df2[keys[0]]).astype(int)
    df2['merge_lon'] = (10000 * df2[keys[1]]).astype(int)
    
    df2.drop(keys, axis=1, inplace=True)
    merged = pd.merge(df1, df2, on=['merge_lat', 'merge_lon'], how=how)
    merged.drop(['merge_lat', 'merge_lon'], axis=1, inplace=True)
    return merged

def create_space(lat, lon, s=10):
    """Creates a s km x s km square centered on (lat, lon)"""
    v = (180/math.pi)*(500/6378137)*s # roughly 0.045 for s=10
    return lat - v, lon - v, lat + v, lon + v
    
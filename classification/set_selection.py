import numpy as np
import pandas as pd

_USEABLE_BLOCKS = [
                    'plate_1x1' ,
                    'plate_1x2' ,
                    'plate_1x3' ,
                    'plate_1x4' ,
                    'plate_1x6' ,
                    'plate_1x8' ,
                    'plate_1x10',
                    'plate_2x2' ,
                    'plate_2x3' ,
                    'plate_2x4' ,
                    'plate_2x6' ,
                    'plate_2x8' ,
                    'plate_2x12',
                    'plate_4x4' ,
                    'plate_4x6' ,
                    'plate_4x8' ,
                    'plate_6x6' ,
                    'plate_6x8' ,
                    'plate_6x10',
                    'plate_8x8' ,
                    'brick_1x1' ,
                    'brick_1x2' ,
                    'brick_1x3' ,
                    'brick_1x4' ,
                    'brick_1x6' ,
                    'brick_1x8' ,
                    'brick_2x2' ,
                    'brick_2x3' ,
                    'brick_2x4' ,
                    'brick_2x6' ,
                  ]

set_df  = pd.read_csv('/home/sean/Insight/legos/classification/data/lego_set_parts.csv'      ).drop(['Unnamed: 0'],axis=1)
part_df = pd.read_csv('/home/sean/Insight/legos/classification/data/individual_set_parts.csv').drop(['Unnamed: 0'],axis=1)
info_df = pd.read_csv('/home/sean/Insight/legos/classification/data/set_name_url.csv'        ).drop(['Unnamed: 0'],axis=1)

def select_valid_sets(
                        inp_dict,
                     ):
    
    possible_set_df = set_df.copy()
    
    # First select where we don't have too many of the known pieces
    for key in inp_dict.keys():
        possible_set_df = possible_set_df[ 
                                            ( possible_set_df[key] <= inp_dict[key] ) 
                                         ]
        
    # The number of parts from the set we have
    # Add this as a column
    possible_set_df['n_have'] = possible_set_df[inp_dict.keys()].sum( axis=1 )

    
    # Then select to make sure we have at least 1 of the known pieces
    possible_set_df = possible_set_df[
                                        ( possible_set_df['n_have'] > 0 )
                                     ]
    
    # Calculate number of parts needed, and fraction had
    possible_set_df['n_needed' ] = possible_set_df['n_parts'] - possible_set_df['n_have' ]
    possible_set_df['frac_have'] = possible_set_df['n_have' ] / possible_set_df['n_parts'].astype(float)
    
    return possible_set_df[
                            ['set_id','set_name','set_url','frac_have','n_parts','n_have','n_needed',]+inp_dict.keys()
                          ].sort_values( ['frac_have','n_parts','n_have'],ascending=[False,True,False]).head(20)
    
    
# Can modify this later
# For now, just take the best
#  when sorted by fraction we
#  have, and the number of parts
def return_rec_set(
                    inp_df,
                  ):
    
    ind = inp_df.index.values[0]
    out_df = inp_df.loc[ind].copy()
    new_ind = info_df['set_id'] == out_df['set_id']
    out_df['set_name'] = info_df.loc[ new_ind, 'set_title'].values[0]
    out_df['set_url' ] = info_df.loc[ new_ind, 'set_url'  ].values[0]
    return out_df
    
    
def get_needed_parts(
                        rec_set,
                        part_dict,
                    ):
    
    all_part_df = part_df
    
    set_id = rec_set['set_id']
    
    set_part_df = all_part_df[ all_part_df['set_id'] == set_id ]
    
    set_part_list = set_part_df['part_name'].unique()
    
    # Get list of all the bricks in the set, and all the plates
    brick_list  = [ block for block in part_dict.keys() if ( 
                                                             ('brick' in block) and 
                                                             (block[0].upper() + block[1:].replace('_',' ').replace('x','X') in set_part_list ) and 
                                                             ( len(block) < 11 ) ) ]
    plate_list  = [ block for block in part_dict.keys() if ( 
                                                             ('plate' in block) and 
                                                             (block[0].upper() + block[1:].replace('_',' ').replace('x','X') in set_part_list ) and 
                                                             ( len(block) < 11 ) ) ]
    other_list  = [ block for block in set_part_df['part_name'].unique() if ( 
                                                                            ( block.replace(' ','_').lower() not in brick_list ) and 
                                                                            ( block.replace(' ','_').lower() not in plate_list ) ) ]
    parts_needed = {}
    parts_have   = {}
    
    
    for part in other_list:
        parts_needed[part] = set_part_df.loc[ set_part_df['part_name']==part ]['quantity'].values[0]
        
    for part_list in [brick_list,plate_list]:
        for part in part_list:
            read_part = part[0].upper() + part[1:].replace('_',' ').replace('x','X')
            parts_have[read_part] = set_part_df.loc[ set_part_df['part_name']==read_part ]['quantity'].values[0]
            
    return parts_have, parts_needed
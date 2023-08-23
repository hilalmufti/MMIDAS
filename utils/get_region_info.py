import numpy as np

def get_roi(df_roi, level=0):
    # level (0): ACA, MOp, SSp, AUD, RSP, PTLp, VIS
    # super regions (I): {MOp,SSp}, {ACA}, {VIS,AUD,PTLp}, {RSP}
    # super regions (II): {MOp,SSp}, {ACA}, {VIS,AUD,PTLp,RSP}
    # super regions (III): {MOp,ACA}, {SSp,PTLp,AUD}, {VIS,RSP}
    # super regions (IV): {MOp,ACA,SSp}, {VIS,AUD,PTLp,RSP}
    # super regions (V): {MOp,ACA,SSp,PTLp,AUD}, {VIS,RSP}
    # super regions (VI): {MOp,ACA,SSp,PTLp}, {VIS,RSP,AUD}

    region_label = df_roi['region_label'].values
    region_id = df_roi['region_id'].values
    region_color = df_roi['region_color'].values

    ssp_idx = np.where(region_label == 'SSp')[0]
    mop_idx = np.where(region_label == 'MOp')[0]
    vis_idx = np.where(region_label == 'VIS')[0]
    aud_idx = np.where(region_label == 'AUD')[0]
    ptlp_idx = np.where(region_label == 'PTLp')[0]
    aca_idx = np.where(region_label == 'ACA')[0]
    rsp_idx = np.where(region_label == 'RSP')[0]


    if level == 1:
        # merging - follow super region (I) - {MOp,ACA}, {AUC}, {SSp,PTLp,RSP}, {VIS}
        region_label[ssp_idx] = region_label[mop_idx[0]]
        region_id[ssp_idx] = region_id[mop_idx[0]]
        region_color[ssp_idx] = region_color[mop_idx[0]]
        region_label[aud_idx] = region_label[vis_idx[0]]
        region_id[aud_idx] = region_id[vis_idx[0]]
        region_color[aud_idx] = region_color[vis_idx[0]]
        region_label[ptlp_idx] = region_label[vis_idx[0]]
        region_id[ptlp_idx] = region_id[vis_idx[0]]
        region_color[ptlp_idx] = region_color[vis_idx[0]]

    elif level == 2:
        # merging - follow super region (VI) - {MOp,ACA,SSp,PTLp,RSP,AUD}, {VIS}
        region_label[aca_idx] = region_label[mop_idx[0]]
        region_id[aca_idx] = region_id[mop_idx[0]]
        region_color[aca_idx] = region_color[mop_idx[0]]
        region_label[ssp_idx] = region_label[mop_idx[0]]
        region_id[ssp_idx] = region_id[mop_idx[0]]
        region_color[ssp_idx] = region_color[mop_idx[0]]
        region_label[aud_idx] = region_label[mop_idx[0]]
        region_id[aud_idx] = region_id[mop_idx[0]]
        region_color[aud_idx] = region_color[mop_idx[0]]
        region_label[ptlp_idx] = region_label[mop_idx[0]]
        region_id[ptlp_idx] = region_id[mop_idx[0]]
        region_color[ptlp_idx] = region_color[mop_idx[0]]
        region_label[rsp_idx] = region_label[mop_idx[0]]
        region_id[rsp_idx] = region_id[mop_idx[0]]
        region_color[rsp_idx] = region_color[mop_idx[0]]

    elif level == 6:
        #merging - follow super region (IV) - {MOp,ACA,SSp}, {VIS,AUD,PTLp,RSP}
        region_label[aca_idx] = region_label[mop_idx[0]]
        region_id[aca_idx] = region_id[mop_idx[0]]
        region_color[aca_idx] = region_color[mop_idx[0]]
        region_label[ssp_idx] = region_label[mop_idx[0]]
        region_id[ssp_idx] = region_id[mop_idx[0]]
        region_color[ssp_idx] = region_color[mop_idx[0]]
        region_label[aud_idx] = region_label[vis_idx[0]]
        region_id[aud_idx] = region_id[vis_idx[0]]
        region_color[aud_idx] = region_color[vis_idx[0]]
        region_label[ptlp_idx] = region_label[vis_idx[0]]
        region_id[ptlp_idx] = region_id[vis_idx[0]]
        region_color[ptlp_idx] = region_color[vis_idx[0]]
        region_label[rsp_idx] = region_label[vis_idx[0]]
        region_id[rsp_idx] = region_id[vis_idx[0]]
        region_color[rsp_idx] = region_color[vis_idx[0]]



    return region_label, region_id, region_color


    # merging - follow super region (II) - {MOp,SSp}, {ACA}, {VIS,AUD,PTLp,RSP}
    # region_label[ssp_idx] = region_label[mop_idx[0]]
    # region_id[ssp_idx] = region_id[mop_idx[0]]
    # region_color[ssp_idx] = region_color[mop_idx[0]]
    # region_label[aud_idx] = region_label[vis_idx[0]]
    # region_id[aud_idx] = region_id[vis_idx[0]]
    # region_color[aud_idx] = region_color[vis_idx[0]]
    # region_label[ptlp_idx] = region_label[vis_idx[0]]
    # region_id[ptlp_idx] = region_id[vis_idx[0]]
    # region_color[ptlp_idx] = region_color[vis_idx[0]]
    # region_label[rsp_idx] = region_label[vis_idx[0]]
    # region_id[rsp_idx] = region_id[vis_idx[0]]
    # region_color[rsp_idx] = region_color[vis_idx[0]]

    # merging - follow super region (III) - {MOp,ACA}, {PTLp,AUD, RSP}, {VIS,SSp}
    # region_label[aca_idx] = region_label[mop_idx[0]]
    # region_id[aca_idx] = region_id[mop_idx[0]]
    # region_color[aca_idx] = region_color[mop_idx[0]]
    # region_label[aud_idx] = region_label[rsp_idx[0]]
    # region_id[aud_idx] = region_id[rsp_idx[0]]
    # region_color[aud_idx] = region_color[rsp_idx[0]]
    # region_label[ptlp_idx] = region_label[rsp_idx[0]]
    # region_id[ptlp_idx] = region_id[rsp_idx[0]]
    # region_color[ptlp_idx] = region_color[ssp_idx[0]]
    # region_label[ssp_idx] = region_label[vis_idx[0]]
    # region_id[ssp_idx] = region_id[vis_idx[0]]
    # region_color[ssp_idx] = region_color[vis_idx[0]]

    # merging - follow super region (IV) - {MOp,ACA,SSp}, {VIS,AUD,PTLp,RSP}
    # region_label[aca_idx] = region_label[mop_idx[0]]
    # region_id[aca_idx] = region_id[mop_idx[0]]
    # region_color[aca_idx] = region_color[mop_idx[0]]
    # region_label[ssp_idx] = region_label[mop_idx[0]]
    # region_id[ssp_idx] = region_id[mop_idx[0]]
    # region_color[ssp_idx] = region_color[mop_idx[0]]
    # region_label[aud_idx] = region_label[vis_idx[0]]
    # region_id[aud_idx] = region_id[vis_idx[0]]
    # region_color[aud_idx] = region_color[vis_idx[0]]
    # region_label[ptlp_idx] = region_label[vis_idx[0]]
    # region_id[ptlp_idx] = region_id[vis_idx[0]]
    # region_color[ptlp_idx] = region_color[vis_idx[0]]
    # region_label[rsp_idx] = region_label[vis_idx[0]]
    # region_id[rsp_idx] = region_id[vis_idx[0]]
    # region_color[rsp_idx] = region_color[vis_idx[0]]

    # merging - follow super region (V) - {MOp,ACA,SSp,PTLp,AUD}, {VIS,RSP}
    # region_label[aca_idx] = region_label[mop_idx[0]]
    # region_id[aca_idx] = region_id[mop_idx[0]]
    # region_color[aca_idx] = region_color[mop_idx[0]]
    # region_label[ssp_idx] = region_label[mop_idx[0]]
    # region_id[ssp_idx] = region_id[mop_idx[0]]
    # region_color[ssp_idx] = region_color[mop_idx[0]]
    # region_label[aud_idx] = region_label[mop_idx[0]]
    # region_id[aud_idx] = region_id[mop_idx[0]]
    # region_color[aud_idx] = region_color[mop_idx[0]]
    # region_label[ptlp_idx] = region_label[mop_idx[0]]
    # region_id[ptlp_idx] = region_id[mop_idx[0]]
    # region_color[ptlp_idx] = region_color[mop_idx[0]]
    # region_label[rsp_idx] = region_label[vis_idx[0]]
    # region_id[rsp_idx] = region_id[vis_idx[0]]
    # region_color[rsp_idx] = region_color[vis_idx[0]]

    # merging - follow super region (VI) - {MOp,ACA,SSp,PTLp}, {VIS,RSP,AUD}
    # region_label[aca_idx] = region_label[mop_idx[0]]
    # region_id[aca_idx] = region_id[mop_idx[0]]
    # region_color[aca_idx] = region_color[mop_idx[0]]
    # region_label[ssp_idx] = region_label[mop_idx[0]]
    # region_id[ssp_idx] = region_id[mop_idx[0]]
    # region_color[ssp_idx] = region_color[mop_idx[0]]
    # region_label[aud_idx] = region_label[vis_idx[0]]
    # region_id[aud_idx] = region_id[vis_idx[0]]
    # region_color[aud_idx] = region_color[vis_idx[0]]
    # region_label[ptlp_idx] = region_label[mop_idx[0]]
    # region_id[ptlp_idx] = region_id[mop_idx[0]]
    # region_color[ptlp_idx] = region_color[mop_idx[0]]
    # region_label[rsp_idx] = region_label[vis_idx[0]]
    # region_id[rsp_idx] = region_id[vis_idx[0]]
    # region_color[rsp_idx] = region_color[vis_idx[0]]


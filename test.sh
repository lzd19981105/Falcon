
# Inference for Image Classification task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/IMG_CLS/[IMG_CLS]_003_AID_3525_river_192_ori.png \
    --post_process_type IMG_CLS \
    --prompt "Classify the image."

# Inference for Visual Question Answering task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/IMG_VQA/[IMG_VQA]_007_HRBEN_5965_1335_ori.png \
    --post_process_type IMG_VQA \
    --prompt "Is the number of roads equal to the number of residential areas?"

# Inference for Counting Target task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/IMG_CT/[IMG_CT]_016_DIOR_25156_13931_ori.png \
    --post_process_type IMG_CT \
    --prompt "Count the number of ship."

# Inference for Image Caption task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/IMG_CAP/[IMG_CAP]_010_RSICD_208_church_56_ori.png \
    --post_process_type IMG_CAP \
    --prompt "Describe the image."

# Inference for Detailed Image Caption task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/IMG_CAP_DETAILED/[IMG_CAP_DETAILED]_026_RSICD_126_commercial_5_ori.png \
    --post_process_type IMG_CAP_DETAILED \
    --prompt "Describe the image in detail."

# Inference for Region Classification-HBB task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/REG_CLS_HBB/[REG_CLS_HBB]_005_DIOR_3829_12264_ori.png \
    --post_process_type REG_CLS_HBB \
    --prompt "Classify the region of <box><855><297><891><355></box>.\nUse one or a few words."

# Inference for Region Classification-OBB task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/REG_CLS_OBB/[REG_CLS_OBB]_001_DIOR_1_11726_ori.png \
    --post_process_type REG_CLS_OBB \
    --prompt "Classify the region of <quad><703><420><703><292><571><292><571><420></quad>.\nUse one or a few words."

# Inference for Region Detection-HBB task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/REG_DET_HBB/[REG_DET_HBB]_004_DIOR_5212_12735_ori.png \
    --post_process_type REG_DET_HBB \
    --prompt "Detect all stadium in the image."

# Inference for Region Detection-OBB task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/REG_DET_OBB/[REG_DET_OBB]_034_DOTA2.0_77716_P0799_ori.png \
    --post_process_type REG_DET_OBB \
    --prompt "Detect all harbor in the image.\nUse oriented bounding boxes."

# Inference for Visual Grounding task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/REG_VG/[REG_VG]_002_DIOR-RSVG_69_00258_ori.png \
    --post_process_type REG_VG \
    --prompt "Detect an area that matches the description.\nfind a swimming pool that is about 118 square meters. there is a parking lot that is about 2988 square meters, located approximately 38 meters northeast of the swimming pool.\nUse horizontal bounding boxes."

# Inference for Region Caption task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/REG_CAP/[REG_CAP]_001_DIOR-RSVG_1_00006_ori.png \
    --post_process_type REG_CAP \
    --prompt "Describe the <box><622><706><696><831></box> in this image."

# Inference for Pixel Classification task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/PIX_CLS/[PIX_CLS]_039_GEONRW_74671_427_5725_rgb_ori.png \
    --post_process_type PIX_CLS \
    --prompt "Classify the region of <poly><1000><0><488><0><465><221><443><279><696><258><704><373><772><343><809><397><631><489><741><704><1000><682><1000><585><965><589><959><509><961><471><1000><413></poly>.\nUse one or a few words."

# Inference for Segmentation task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/PIX_SEG/[PIX_SEG]_034_GEONRW_376_5755_rgb-ori.png \
    --post_process_type PIX_SEG \
    --prompt "Segment out road in the image."

# Inference for Change Detection task
python inference.py \
    --checkpoint_path <path_to_the_checkpoint_you_want> \
    --image_path image_samples/PIX_CHG/[PIX_CHG]_199_WHU-CD_28911_590_ori.png \
    --image2_path image_samples/PIX_CHG/[PIX_CHG]_199_WHU-CD_28911_590_post.png \
    --post_process_type PIX_CHG \
    --prompt "Find changes in the two images."

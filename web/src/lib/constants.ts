export const CLASS_NAMES = [
  'Corn_Cercospora_Gray_leaf_spot',
  'Corn_Common_rust',
  'Corn_Northern_Leaf_Blight',
  'Corn_healthy',
  'Potato_Early_blight',
  'Potato_Late_blight',
  'Potato_healthy',
  'Tomato_Bacterial_spot',
  'Tomato_Early_blight',
  'Tomato_Late_blight',
  'Tomato_Leaf_Mold',
  'Tomato_Septoria_leaf_spot',
  'Tomato_Spider_mites',
  'Tomato_Target_Spot',
  'Tomato_Yellow_Leaf_Curl_Virus',
  'Tomato_mosaic_virus',
  'Tomato_healthy',
] as const;

export type ClassName = typeof CLASS_NAMES[number];

export const INPUT_H = 224;
export const INPUT_W = 224;
export const N_CLASSES = 17;

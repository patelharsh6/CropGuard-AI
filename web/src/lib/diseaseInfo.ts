import { ClassName } from './constants';

export interface DiseaseInfo {
  displayName: string;
  cause: string;
  prevention: string[];
  treatmentCategory: string;
  referenceUrl: string;
}

export const DISEASE_INFO: Record<ClassName, DiseaseInfo> = {
  Corn_Cercospora_Gray_leaf_spot: {
    displayName: "Corn Gray Leaf Spot",
    cause: "Fungal pathogen (Cercospora zeae-maydis). Spores overwinter in crop residue and thrive in prolonged periods of high humidity and overcast days.",
    prevention: [
      "Practice crop rotation to reduce inoculum in the soil.",
      "Use conservation tillage practices carefully, as burying residue reduces spore survival.",
      "Plant resistant or tolerant corn hybrids.",
    ],
    treatmentCategory: "Foliar fungicide application. Timing is critical, usually applied at the tasseling to early silking stages if disease pressure is high.",
    referenceUrl: "https://extension.psu.edu/corn-disease-gray-leaf-spot",
  },
  Corn_Common_rust: {
    displayName: "Corn Common Rust",
    cause: "Fungal pathogen (Puccinia sorghi). Spores are blown in from southern climates; development is favored by cool, moist weather.",
    prevention: [
      "Plant rust-resistant sweet corn varieties.",
      "Plant early in the season to avoid peak spore showers in late summer.",
    ],
    treatmentCategory: "Fungicide application. Treatment is rarely economically justified for field corn but may be necessary for highly susceptible sweet corn varieties if symptoms appear before silking.",
    referenceUrl: "https://extension.umn.edu/corn-pest-management/common-rust-corn",
  },
  Corn_Northern_Leaf_Blight: {
    displayName: "Corn Northern Leaf Blight",
    cause: "Fungal pathogen (Exserohilum turcicum). Overwinters in corn residue; favored by moderate temperatures and extended periods of leaf wetness.",
    prevention: [
      "Select resistant corn hybrids (look for major gene resistance like Ht1, Ht2, Ht3, or HtN).",
      "Manage crop residue through tillage or rotation to non-host crops.",
    ],
    treatmentCategory: "Foliar fungicides. Most effective when applied at or shortly after tasseling when the disease is actively spreading to the upper canopy.",
    referenceUrl: "https://crop-protection-network.s3.amazonaws.com/publications/cpn-2002-northern-corn-leaf-blight.pdf",
  },
  Corn_healthy: {
    displayName: "Healthy Corn",
    cause: "Optimal environmental conditions, adequate soil fertility, and an absence of significant pathogens or pests.",
    prevention: [
      "Plant seeds in blocks (at least 3-4 rows side-by-side) rather than single long rows to ensure proper wind pollination.",
      "Maintain consistent soil moisture, providing approximately 1 inch of water per week, especially critical during silking.",
      "Side-dress with nitrogen fertilizer when plants are 6-12 inches tall.",
    ],
    treatmentCategory: "Continue current care practices. Monitor regularly for common pests like the corn earworm or signs of nutrient deficiency.",
    referenceUrl: "https://extension.usu.edu/yardandgarden/research/sweet-corn-in-the-garden",
  },
  Potato_Early_blight: {
    displayName: "Potato Early Blight",
    cause: "Fungal pathogen (Alternaria solani). Overwinters in soil and plant debris; primarily affects older foliage undergoing stress or natural senescence.",
    prevention: [
      "Maintain proper fertility, especially nitrogen, to keep plants vigorous.",
      "Rotate crops with non-solanaceous hosts for at least 2-3 years.",
      "Allow potato tubers to fully mature and skins to set before harvesting to prevent tuber infection.",
    ],
    treatmentCategory: "Apply protectant fungicides preventatively or at the first sign of lesions on lower leaves.",
    referenceUrl: "https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
  },
  Potato_Late_blight: {
    displayName: "Potato Late Blight",
    cause: "Oomycete pathogen (Phytophthora infestans). A highly destructive disease favored by cool, wet weather. Survives in living tissue, such as volunteer potatoes or cull piles.",
    prevention: [
      "Plant certified disease-free seed potatoes.",
      "Destroy cull piles and control volunteer potato plants early in the season.",
      "Monitor weather-based blight forecasting systems to time preventative sprays.",
    ],
    treatmentCategory: "Aggressive, frequent fungicide applications (protectant and systemic depending on severity). Infected plants should be immediately removed and destroyed to prevent spread.",
    referenceUrl: "https://extension.psu.edu/potato-late-blight",
  },
  Potato_healthy: {
    displayName: "Healthy Potato",
    cause: "Good soil structure, appropriate moisture levels, and freedom from disease and pests.",
    prevention: [
      "Hill up soil around the stems as they grow to protect developing tubers from sunlight (which causes greening) and blight spores.",
      "Maintain even soil moisture to prevent tuber disorders like hollow heart or knobby growth.",
      "Harvest after vines have naturally died back and tuber skins are thick.",
    ],
    treatmentCategory: "Continue monitoring for pests like the Colorado potato beetle and ensure soil drains well to prevent tuber rot.",
    referenceUrl: "https://extension.unh.edu/resource/growing-potatoes-fact-sheet",
  },
  Tomato_Bacterial_spot: {
    displayName: "Tomato Bacterial Spot",
    cause: "Bacterial pathogen (Xanthomonas species). Survives on seeds, volunteer tomatoes, and infected plant debris. Spreads rapidly through splashing rain or overhead irrigation.",
    prevention: [
      "Purchase certified disease-free seeds and transplants.",
      "Avoid working in the garden when foliage is wet to prevent mechanical spread.",
      "Practice strict sanitation and a 3-year crop rotation.",
    ],
    treatmentCategory: "Chemical control is difficult. Copper-based bactericides can be applied preventatively to slow the spread, but are ineffective once an epidemic is established.",
    referenceUrl: "https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper",
  },
  Tomato_Early_blight: {
    displayName: "Tomato Early Blight",
    cause: "Fungal pathogen (Alternaria solani). It overwinters in infected plant debris and soil, spreading most rapidly during periods of high humidity, heavy dew, and frequent rain.",
    prevention: [
      "Rotate crops (wait 3-4 years before planting nightshade family crops in the same spot).",
      "Stake or cage plants to improve airflow and keep foliage off the ground.",
      "Apply organic mulch to prevent soil spores from splashing onto lower leaves.",
      "Use drip irrigation or water at the base to keep foliage dry.",
    ],
    treatmentCategory: "Remove and destroy severely infected lower leaves. Apply protectant fungicides (such as copper-based or chlorothalonil products) preventatively or at the very first sign of disease.",
    referenceUrl: "https://extension.colostate.edu/topic-areas/yard-garden/early-blight-of-potato-and-tomato-2-926/",
  },
  Tomato_Late_blight: {
    displayName: "Tomato Late Blight",
    cause: "Oomycete pathogen (Phytophthora infestans). Rapidly destroys plants under cool, continuously wet conditions. Spores can blow in from miles away.",
    prevention: [
      "Grow late blight-resistant tomato varieties if the disease is common in your area.",
      "Maximize airflow around plants and ensure foliage dries quickly.",
      "Monitor local agricultural alerts for late blight outbreaks.",
    ],
    treatmentCategory: "Protectant fungicides are required before infection occurs. Once infected, plants cannot be saved and must be bagged and disposed of in the trash to protect neighboring crops.",
    referenceUrl: "https://hort.extension.wisc.edu/articles/late-blight/",
  },
  Tomato_Leaf_Mold: {
    displayName: "Tomato Leaf Mold",
    cause: "Fungal pathogen (Passalora fulva). A significant problem in high tunnels and greenhouses where humidity is high (>85%) and air circulation is poor.",
    prevention: [
      "Improve air circulation by pruning lower leaves and spacing plants adequately.",
      "Ventilate greenhouses and high tunnels aggressively to lower humidity.",
      "Grow resistant varieties (though new fungal races frequently overcome resistance).",
    ],
    treatmentCategory: "Fungicide application. Focus on cultural controls (lowering humidity) first, as chemicals are less effective if the environment remains conducive to the fungus.",
    referenceUrl: "https://extension.umn.edu/disease-management/tomato-leaf-mold",
  },
  Tomato_Septoria_leaf_spot: {
    displayName: "Tomato Septoria Leaf Spot",
    cause: "Fungal pathogen (Septoria lycopersici). Very common foliar disease; overwinters on solanaceous weeds and tomato debris, splashing onto lower leaves during rain.",
    prevention: [
      "Mulch heavily around the base of the plant to create a physical barrier against soil-borne spores.",
      "Water at the base of the plant only; keep foliage dry.",
      "Control solanaceous weeds (like horsenettle or nightshade) near the garden.",
    ],
    treatmentCategory: "Remove affected lower leaves immediately. Apply protectant fungicides preventatively on a 7-10 day schedule during wet weather.",
    referenceUrl: "https://extension.psu.edu/septoria-leaf-spot-on-tomatoes",
  },
  Tomato_Spider_mites: {
    displayName: "Tomato Two-Spotted Spider Mites",
    cause: "Arachnid pests (Tetranychus urticae). Mites multiply exponentially during hot, dry, dusty weather, piercing plant cells and sucking out contents.",
    prevention: [
      "Keep plants well-watered to reduce drought stress, which makes them highly susceptible.",
      "Periodically wash foliage down with a strong stream of water to dislodge mites and reduce dust.",
      "Encourage natural predators like lady beetles and predatory mites.",
    ],
    treatmentCategory: "Horticultural oils, insecticidal soaps, or specific miticides. Ensure thorough coverage of the undersides of leaves where mites congregate. Avoid broad-spectrum insecticides which kill natural predators.",
    referenceUrl: "https://ipm.ucanr.edu/agriculture/tomato/spider-mites/",
  },
  Tomato_Target_Spot: {
    displayName: "Tomato Target Spot",
    cause: "Fungal pathogen (Corynespora cassiicola). Thrives in high humidity and requires long periods of leaf wetness to infect, spreading via wind and splashing rain.",
    prevention: [
      "Prune plants to improve air circulation and reduce humidity within the canopy.",
      "Avoid overhead watering and handle plants only when foliage is completely dry.",
      "Remove and destroy all plant debris at the end of the season, as the fungus survives on refuse.",
    ],
    treatmentCategory: "Preventative fungicide application. Note that resistance to some systemic chemical classes is emerging, making rotation of active ingredients crucial if relying on chemical control.",
    referenceUrl: "https://vegcropshotline.org/article/target-spot-of-tomato/",
  },
  Tomato_Yellow_Leaf_Curl_Virus: {
    displayName: "Tomato Yellow Leaf Curl Virus (TYLCV)",
    cause: "Viral pathogen transmitted exclusively by the silverleaf whitefly (Bemisia tabaci). The virus prevents fruit set and severely stunts plant growth.",
    prevention: [
      "Plant TYLCV-resistant tomato varieties.",
      "Use reflective mulches to repel whiteflies early in the season.",
      "Protect young transplants with fine-mesh row covers until they begin flowering.",
    ],
    treatmentCategory: "There is no cure for virus-infected plants. Infected plants should be uprooted and destroyed immediately. Vector control (managing whitefly populations) is the only management strategy.",
    referenceUrl: "https://edis.ifas.ufl.edu/publication/IN1146",
  },
  Tomato_mosaic_virus: {
    displayName: "Tomato Mosaic Virus (ToMV)",
    cause: "Viral pathogen highly related to Tobacco Mosaic Virus (TMV). Extremely stable and highly contagious; spreads mechanically via contaminated hands, tools, or clothing.",
    prevention: [
      "Wash hands thoroughly with soap and water before handling tomato plants, especially if you use tobacco products.",
      "Disinfect garden tools frequently (e.g., with a 10% bleach solution).",
      "Do not save seeds from infected plants.",
    ],
    treatmentCategory: "No cure exists. Infected plants must be completely removed (including roots) and destroyed. Do not compost infected material, as the virus can survive for years.",
    referenceUrl: "https://extension.umn.edu/disease-management/tomato-viruses",
  },
  Tomato_healthy: {
    displayName: "Healthy Tomato",
    cause: "Excellent cultural practices, proper spacing, and proactive disease prevention.",
    prevention: [
      "Plant in full sun (at least 6-8 hours daily) and ensure soil is well-drained and rich in organic matter.",
      "Provide sturdy support (stakes or cages) and prune suckers to maintain manageable growth and airflow.",
      "Water deeply and consistently at the soil level to prevent blossom end rot and foliar diseases.",
    ],
    treatmentCategory: "Keep up the good work. Apply a balanced fertilizer according to soil test recommendations once fruits begin to set.",
    referenceUrl: "https://extension.unh.edu/resource/growing-tomatoes-fact-sheet",
  }
};

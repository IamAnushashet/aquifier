import numpy as np
import pandas as pd
from typing import Dict, List, Any

class RechargeRecommendationEngine:
    """
    Dynamic AI Water Management and Recharge Decision Engine.
    Generates tailored civil engineering, agronomic, and policy recommendations
    contingent on:
    - Predicted groundwater table depth
    - Predicted decline trajectory
    - Upcoming rainfall forecast
    - Aquifer geological formation type
    - Extraction pressure
    """
    
    @staticmethod
    def generate_recommendations(
        well_meta: Dict[str, Any],
        current_depth_m: float,
        predicted_depth_30d_m: float,
        predicted_rain_30d_mm: float,
        extraction_rate: float,
        risk_category: str
    ) -> Dict[str, Any]:
        """
        Dynamically constructs actionable, site-specific recharge & conservation interventions.
        """
        recommendations = []
        aquifer_type = well_meta.get("aquifer_type", "Unconfined Alluvial")
        critical_depth = well_meta.get("critical_depth_m", 28.0)
        safe_depth = well_meta.get("safe_depth_m", 14.0)
        decline_m = predicted_depth_30d_m - current_depth_m
        
        # 1. Pumping & Extraction Quota Recommendation
        if decline_m > 0.6 or risk_category in ["High Risk", "Critical Risk"]:
            reduction_target = 30 if risk_category == "Critical Risk" else (20 if decline_m > 0.8 else 15)
            recommendations.append({
                "action": "Managed Groundwater Extraction Curtailment",
                "category": "Policy & Demand Management",
                "priority": "HIGH",
                "urgency_badge": "Immediate Implementation",
                "description": f"Predicted groundwater decline is steep (+{round(decline_m, 2)}m over 30 days). Enforce a mandatory {reduction_target}% reduction in industrial and tubewell withdrawal.",
                "expected_impact": f"Preserves approximately {round(reduction_target * 0.045, 2)}m of head drawdown over the subsequent cycle."
            })
        elif extraction_rate > 1.2:
            recommendations.append({
                "action": "Staggered Micro-Irrigation Scheduling",
                "category": "Demand Management",
                "priority": "MEDIUM",
                "urgency_badge": "Next Crop Cycle",
                "description": "Adopt precision drip and sprinkler systems to curtail agricultural withdrawal by 10-15%.",
                "expected_impact": "Reduces peak extraction drawdown cones by 18%."
            })

        # 2. Structural Recharge Interventions (Tied to Geology & Upcoming Rainfall)
        if predicted_rain_30d_mm > 50.0:
            # Substantial rainfall expected: prime the recharge capture!
            if "Alluvial" in aquifer_type or "Sand" in aquifer_type:
                recommendations.append({
                    "action": "Percolation Ponds & Infiltration Basins",
                    "category": "Managed Aquifer Recharge (MAR)",
                    "priority": "HIGH",
                    "urgency_badge": "Pre-Monsoon Window",
                    "description": f"With {round(predicted_rain_30d_mm, 1)}mm precipitation forecasted, excavate high-permeability percolation ponds to capture surface runoff into the sandy vadose zone.",
                    "expected_impact": "Directly enhances aquifer recharge by +35% to 45% of captured stormwater."
                })
                recommendations.append({
                    "action": "Urban Rooftop Rainwater Harvesting (RWH)",
                    "category": "Urban Infiltration",
                    "priority": "HIGH",
                    "urgency_badge": "Mandatory Structural",
                    "description": "Mandate recharge pits connected to rooftop catchment filters across industrial and municipal buildings.",
                    "expected_impact": "Alleviates urban storm runoff and feeds unconfined aquifer layers."
                })
            elif "Crystalline" in aquifer_type or "Fractured" in aquifer_type:
                recommendations.append({
                    "action": "Recharge Shafts & Gravity Injection Wells",
                    "category": "Deep Aquifer MAR",
                    "priority": "HIGH",
                    "urgency_badge": "Immediate",
                    "description": f"Given hard-rock crystalline geology with low surface permeability, install deep recharge borewells and desilting chambers to bypass upper clays.",
                    "expected_impact": "Direct deep recharge transmission efficiency exceeding 60%."
                })
                recommendations.append({
                    "action": "Continuous Contour Trenching & Check Dams",
                    "category": "Catchment Hydrology",
                    "priority": "MEDIUM",
                    "urgency_badge": "Watershed Scale",
                    "description": "Construct check dams along natural stream order tributaries to decelerate runoff velocity and enhance local infiltration time.",
                    "expected_impact": "Extends surface ponding retention time from 4 days to 28 days."
                })
        else:
            # Low rainfall forecast (Dry spell)
            recommendations.append({
                "action": "Treated Effluent Subsurface Infiltration (Recycling)",
                "category": "Water Recycling",
                "priority": "HIGH",
                "urgency_badge": "Dry Season Protocol",
                "description": f"Predicted rainfall is minimal ({round(predicted_rain_30d_mm, 1)}mm). Utilize tertiary-treated wastewater for artificial basin percolation to maintain aquifer baseline pressure.",
                "expected_impact": "Maintains hydraulic head and mitigates subsidence or saltwater intrusion."
            })
            recommendations.append({
                "action": "Dry-Well Desilting and Rehabilitation",
                "category": "Maintenance",
                "priority": "MEDIUM",
                "urgency_badge": "Maintenance Window",
                "description": "Desilt existing recharge pits and injection wells during the dry period before monsoon onset to ensure optimal infiltration rates.",
                "expected_impact": "Restores percolation velocity up to 2.4 m/day."
            })

        # 3. Overall Strategic Guidance Synthesis
        synthesis = (
            f"Groundwater situation is classified as '{risk_category}'. "
            f"With current depth at {round(current_depth_m, 2)}m and a projected 30-day level of {round(predicted_depth_30d_m, 2)}m, "
            f"the primary AI recommendation is to combine {recommendations[0]['action']} with "
            f"{recommendations[1]['action'] if len(recommendations) > 1 else 'continuous telemetry monitoring'}."
        )

        return {
            "well_id": well_meta.get("well_id"),
            "well_name": well_meta.get("name"),
            "risk_category": risk_category,
            "executive_recommendation_quote": synthesis,
            "actions_list": recommendations,
            "contextual_inputs": {
                "current_level_m": current_depth_m,
                "projected_level_m": predicted_depth_30d_m,
                "projected_rainfall_mm": predicted_rain_30d_mm,
                "aquifer_type": aquifer_type
            }
        }

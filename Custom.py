from optparse import Option
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")

# 1. load enviroment 
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    
# 2. pre-process: transfer into JSON    
def input_processor(user_raw_input):
    """
    pre-process: transfer into JSON
    """
    prompt = (
        "You are an intent recognition assistant. Please convert the user's vague requirements into the following specific key "
        "word and in JSON format:\n"
        "{\"Context/Scenario\", \"Given Parameters\", \"The Stem\", \"Options\"}\n"
        f"User Input：'{user_raw_input}'\n"
        "Please return only the JSON, without any explanation."
    )
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={'type': 'json_object'} # return a JSON object directly
    )
    return json.loads(response.choices[0].message.content) 



def get_custom_response(Structured_input):

    # 3. Instruction
    CUSTOM_INSTRUCTION = (
        "##role:\n"
        "You are a senior expert in the field of HVAC (heating, ventilation, and air conditioning) industry,"
        " and a specialist for the Certified HVAC Designer (CHD) exam. Your expertise includes ASHRAE" 
        "standards, fluid dynamics, thermodynamics, and psychrometrics. And you are going to do the single-choice question in the exam which the question is in ASHRAE Terminology.\n"

        "##Knowledge Base:\n"
        "Information extracted from <ASHRAE Handbook - Fundmentals> and other related HVAC reference materials.\n"

        "##Objective:\n"
        "Analyze the provided structured HVAC exam question. You must ensure the logical reasoning "
        "(justification) is physically accurate and directly supports the final selected answer. Output formation should be similar to the examples provided\n"


        "##Mandatory Reasoning Protocol:\n"
        "When faced with technical questions, you must remember question is HVAC field terminology(ASHRAE Terminology) and solve the problem through structural and functional analysis as belowing:\n"
        "Phase 1: Variable & Constraint Identification:Extract every technical parameter and physical constraint\n"
        "Phase 2: Scientific/Engineering Principle Alignment:Identify the specific HVAC principle. Differential Check: Contrast the target concept with its functional opposite.\n"
        "Phase 3: Boundary Condition & 'Industry Context' Check :Identify the technical level of the question (Vocational/Intro vs. Advanced Physics).Check for 'Distractors' that are physically true but contextually irrelevant.  Identify if the question tests the historical/standard origin of a unit. If the physical constants are present and accurately described, the statement should be considered True in an industry certification context, even if the phrasing is slightly archaic or less common in modern digital controls. Remember the question is HVAC field terminology(ASHRAE Terminology)\n"
        "Phase 4: Independent Option Evaluation: Evaluate each option (A, B, C, D) independently. Definition Filter: Is the option a property of the concept, or its defining characteristic? Determine its physical validity without referencing other options.\n"
        "Phase 5: Internal Consistency Check:The 'Indicator over Category' Rule: In HVAC, how we measure or sense a change (Thermometer, Pressure Gauge, Sight Glass) is usually the 'Correct' answer for a definition question, as it relates to field diagnostics.Logical Loop-back: Does the selected answer explain how a technician or a controller distinguishes this state from another?\n" 
        "Phase 6: Final Conclusion: State the only one correct option based strictly on the verified logic above.\n"
        
        "##Constraint:\n"
        "If no answer is found in the source, only then use general knowledge.And select only one answer.\n"
        "Check for Physical Essence over Nomenclature: In T/F questions, if the physical constants and conditions (e.g., mass, temperature, time) perfectly align with a standard HVAC principle, the statement should be judged as True, even if the terminology used is historical, non-standard, or slightly archaic.\n"

        )
    
    # 4. Few-shot
    Example_1_Input = {
            "question": (
                "A planned renovation of a $2,410 \\text{ ft}^2$ ($224 \\text{ m}^2$) retail space "
                "with a 40-occupant capacity is located on the second floor of a 5-story commercial building. "
                "Calculate the approximate minimum outdoor air requirement if the people outdoor air rate is "
                "$7.5 \\text{ cfm/person}$ ($3.8 \\text{ l/s-person}$) and the area outdoor air rate "
                "$0.18 \\text{ cfm/ft}^2$ ($0.9 \\text{ l/s-m2}$)."
           ),
            "options": [
                "A. $489 \\text{ cfm}$ ($230.8 \\text{ l/s}$)",
                "B. $689 \\text{ cfm}$ ($325.3 \\text{ l/s}$)",
                "C. $734 \\text{ cfm}$ ($353.6 \\text{ l/s}$)"
            ]
        }
    
    Example_1_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- Total Area ($A$): $2,410 \\text{ ft}^2$ ($224 \\text{ m}^2$)\n"
            "- Occupancy ($P$): 40 people\n"
            "- People Outdoor Air Rate ($R_p$): $7.5 \\text{ cfm/person}$ ($3.8 \\text{ l/s-person}$)\n"
            "- Area Outdoor Air Rate ($R_a$): $0.18 \\text{ cfm/ft}^2$ ($0.9 \\text{ l/s-m}^2$)\n"
            "- Building Context: 2nd floor of a 5-story commercial building (Retail space renovation).\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "According to ASHRAE Standard 62.1 (Ventilation for Acceptable Indoor Air Quality), the Breathing Zone Outdoor Airflow ($V_{bz}$) for a single zone is calculated using the Ventilation Rate Procedure: $V_{bz} = (R_p \\times P) + (R_a \\times A)$.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- Distractor: The '5-story building' and '2nd floor' location are contextual details but do not alter the fundamental $V_{bz}$ calculation for this specific retail space.\n"
            "- Unit Consistency: Calculations must be performed in Imperial (cfm) and Metric (l/s) units separately to ensure alignment with options.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: $489 \\text{ cfm}$ ($230.8 \\text{ l/s}$). Mathematically inconsistent with the sum of people and area rates.\n"
            "- Option B: $689 \\text{ cfm}$ ($325.3 \\text{ l/s}$). Incorrect sum; potentially ignores or miscalculates one of the core components.\n"
            "- Option C: $734 \\text{ cfm}$ ($353.6 \\text{ l/s}$).\n"
            "  * Imperial: $(7.5 \\times 40) + (0.18 \\times 2,410) = 300 + 433.8 = 733.8 \\text{ cfm} \\approx 734 \\text{ cfm}$.\n"
            "  * Metric: $(3.8 \\times 40) + (0.9 \\times 224) = 152 + 201.6 = 353.6 \\text{ l/s}$.\n"
            "  Physically and mathematically valid.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The reasoning identifies that both people-based and area-based ventilation requirements must be summed. The sum ($733.8 \\text{ cfm}$) aligns perfectly with the rounded value in Option C.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: C. $734 \\text{ cfm}$ ($353.6 \\text{ l/s}$)"
        ) 

    Example_2_Input = {
            "question": (
                "There are three basic types of variable refrigerant flow (VRF) systems: cooling only, heat pump, and ___________."
           ),
            "options": [
                "A. air-cooled outdoor",
                "B. heat recovery",
                "C. water-cooled outdoor"
            ]
        }
    
    Example_2_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- System Type: Variable Refrigerant Flow (VRF) systems.\n"
            "- Classification Criteria: Functional operation modes (Cooling only, Heat pump, and the missing third category).\n"
            "- Primary Constraint: Identify the fundamental architectural type of VRF based on thermodynamic capability rather than heat rejection media.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "VRF systems are categorized by their ability to manage thermal energy across multiple zones. \n"
            "1. Cooling Only: Unidirectional heat removal.\n"
            "2. Heat Pump: Reversible cycle (heating or cooling) for all connected indoor units globally.\n"
            "3. Simultaneous Operation: The thermodynamic principle of shifting energy from a cooling-load zone to a heating-load zone within the same refrigerant circuit.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- Distractor Check: 'Air-cooled' and 'Water-cooled' refer to the heat exchange medium at the condenser (source/sink side), not the functional logic of the refrigerant distribution (load side). These are sub-categories of hardware, not the 'basic type' of cycle logic requested.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'air-cooled outdoor'. This describes a cooling medium for the outdoor unit, not a system-level operational mode.\n"
            "- Option B: 'heat recovery'. This defines a specific VRF topology (3-pipe or 2-pipe with branch controllers) that allows for simultaneous heating and cooling by recovering waste heat. This completes the functional triad of VRF systems.\n"
            "- Option C: 'water-cooled outdoor'. Similar to Option A, this is a hardware configuration for heat rejection, not a basic functional type of the VRF cycle.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The classification of 'Cooling Only', 'Heat Pump', and 'Heat Recovery' covers all possible thermodynamic states of multi-zone refrigerant distribution. Adding a heat rejection method (A or C) would create a category error in the list.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: B. heat recovery"
        )
    
    Example_3_Input = {
            "question": (
                "Which of the following statements is MOST accurate regarding the Task of sequencing heating and cooling?"
           ),
            "options": [
                "A. Central fan systems should not use cool outdoor air in sequence between heating and cooling.",
                "B. Heating and cooling should be supplied simultaneously for humidity control.",
                "C. Zoning and system selection should eliminate, or at least minimize, simultaneous heating and cooling."
            ]
        }
    
    Example_3_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- Task: Sequencing of heating and cooling in HVAC systems.\n"
            "- Goal: Identify the most accurate operational or design strategy.\n"
            "- Physical Constraints: Energy efficiency, thermodynamic logic, and system control stability.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "The Second Law of Thermodynamics and energy conservation principles (such as ASHRAE 90.1) dictate that simultaneous heating and cooling (reheating or cross-mixing) represent a parasitic energy loss. Systems should be sequenced to provide a 'deadband'—a temperature range where neither heating nor cooling is active—to prevent oscillatory control and energy waste.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- 'Humidity Control' (Option B) is a common distractor. While reheating is a physical method for dehumidification, it is an energy-intensive process that modern codes (like ASHRAE 90.1) strictly limit or require specific energy recovery for, rather than recommending it as a general 'sequencing' rule.\n"
            "- 'Free Cooling' (Option A) is a valid operational state but the statement 'should not use' is a false constraint; economizers are actually mandated in many climates.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'Central fan systems should not use cool outdoor air...' Inaccurate. Air-side economizers are a standard efficiency requirement to provide 'free cooling' between mechanical cooling and heating modes.\n"
            "- Option B: 'Heating and cooling should be supplied simultaneously...' Inaccurate. This describes a 'fighting' system. Except for specialized precision dehumidification, this violates energy codes and efficiency standards.\n"
            "- Option C: 'Zoning and system selection should eliminate, or at least minimize, simultaneous heating and cooling.' Accurate. This is the primary objective of HVAC design. Proper zoning ensures that diverse loads are handled independently without thermal conflict.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The selection of Option C aligns with the fundamental engineering goal of minimizing entropy production and maximizing system efficiency through proper spatial (zoning) and mechanical (system selection) design.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: C. Zoning and system selection should eliminate, or at least minimize, simultaneous heating and cooling."
        )
    
    Example_4_Input = {
            "question": (
                "What is heat, which causes a change in the state of a material without a change in temperature, called?"
           ),
            "options": [
                "A. Latent heat",
                "B. Sensible heat",
                "C. Superheat",
                "D. Regular heat"
            ]
        }
    
    Example_4_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- Phenomenon: Heat transfer to/from a material.\n"
            "- Constraint 1: Change in state (phase change) must occur.\n"
            r"- Constraint 2: Temperature remains constant ($\Delta T = 0$) during the process.\n"
            "- Goal: Identify the specific thermodynamic term for this type of energy.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "Thermodynamically, heat addition can manifest in two ways: changing the kinetic energy of molecules (temperature change) or changing the potential energy/molecular bonds (phase change). \n"
            r"- The equation $Q = mc\Delta T$ defines Sensible Heat (temperature change).\n"
            "- The equation $Q = mL$ defines Latent Heat (phase change at constant temperature/pressure).\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            r"- 'Superheat' (Option C) is a 'false friend' because while it involves phase-specific states, it refers to the temperature increase of a vapor *after* it has already completed its phase change from a liquid. It inherently involves a $\Delta T > 0$.\n"
            "- 'Regular heat' (Option D) is non-technical vernacular and lacks scientific definition in engineering.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'Latent heat'. Strictly defined as the energy absorbed or released during a constant-temperature phase change (e.g., fusion or vaporization). This matches all constraints.\n"
            "- Option B: 'Sensible heat'. Defined as heat that results in a measurable temperature change. This violates Constraint 2.\n"
            "- Option C: 'Superheat'. Refers to heating a gas above its saturation temperature. This violates Constraint 2.\n"
            "- Option D: 'Regular heat'. This is not a standardized thermodynamic term.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The reasoning confirms that 'Latent' (meaning 'hidden') refers specifically to energy that does not manifest as a temperature rise on a thermometer but instead drives a molecular structural change. This is internally consistent with the question's premise.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: A. Latent heat"
        )

    Example_5_Input = {
            "question": (
                "What is a sling psychrometer used to measure?"
           ),
            "options": [
                "A. Latent heat",
                "B. Super heat",
                "C. Wet and dry bulb temperature",
                "D. Barometric pressure"
            ]
        }
    
    Example_5_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- Instrument: Sling Psychrometer.\n"
            "- Components: Two matched thermometers (one dry, one with a wetted wick/sock).\n"
            "- Operating State: Manual rotation/slinging to create airflow ($>3 \\text{ m/s}$) across the bulbs.\n"
            "- Goal: Identify the direct physical parameters measured by this specific hardware.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "The device operates on the principle of Evaporative Cooling. \n"
            "1. The dry bulb measures the ambient air temperature ($T_{db}$).\n"
            "2. The wet bulb measures the temperature reached by the evaporation of water into the air stream ($T_{wb}$).\n"
            "The rate of evaporation (and thus the cooling effect) is a function of the air's moisture content, but the instrument itself directly reads temperatures, not derived values like enthalpy or humidity.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- 'Relative Humidity' and 'Latent Heat' are derived parameters. While a psychrometer is used to *find* them via a psychrometric chart, it does not *measure* them directly.\n"
            "- 'Wet bulb' temperature can never exceed 'Dry bulb' temperature; at $100\\%$ saturation, they are equal ($T_{wb} = T_{db}$).\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'Latent heat'. This is a thermodynamic property of a substance's state change, not a direct measurement of a psychrometer.\n"
            "- Option B: 'Super heat'. This refers to the temperature of a vapor above its boiling point, measured with pressure gauges and temperature probes in refrigeration cycles.\n"
            "- Option C: 'Wet and dry bulb temperature'. This is the fundamental data produced by the two thermometers on the device. Mathematically valid as the primary output.\n"
            "- Option D: 'Barometric pressure'. Measured by a barometer, utilizing the weight of the atmosphere, which is unrelated to the psychrometric slinging process.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The reasoning confirms that the hardware configuration (two thermometers, one wetted) is specifically designed to isolate the dry-bulb and wet-bulb temperatures through convective heat transfer and evaporation. This aligns perfectly with Option C.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: C. Wet and dry bulb temperature"
        )
    
    Example_6_Input = {
            "question": (
                "Heat which causes a change in temperature of a substance is called:"
           ),
            "options": [
                "A. Latent heat",
                "B. sensible heat",
                "C. superheat",
                "D. regular heat"
            ]
        }
    
    Example_6_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- Phenomenon: Heat transfer to/from a substance.\n"
            r"- Primary Constraint: The heat results in a measurable change in temperature ($\Delta T \\neq 0$).\n"
            "- Secondary Constraint: The substance does not undergo a phase change during this specific energy exchange.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "In thermodynamics, the total enthalpy change of a substance can be split into sensible and latent components. \n"
            "Sensible heat is defined by the relationship $Q = m \\cdot c_p \\cdot \\Delta T$. The term 'sensible' originates from the fact that the change can be 'sensed' or measured by a thermometer as a change in the kinetic energy of the molecules.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            r"- 'Latent Heat' is the primary distractor; it involves heat transfer where $\Delta T = 0$ but a phase change occurs.\n"
            "- 'Superheat' is a specific application of sensible heat in refrigeration (heating vapor above its saturation point), but it is a sub-category rather than the fundamental name of the heat type itself.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'Latent heat'. Physically invalid here as it specifically refers to heat that does NOT cause a temperature change during a state transition.\n"
            "- Option B: 'sensible heat'. Physically valid. It is defined specifically as the energy required to change the temperature of a substance without changing its phase.\n"
            "- Option C: 'superheat'. While it involves a temperature change, it is a specific state of a refrigerant vapor, not the general name for temperature-changing heat.\n"
            "- Option D: 'regular heat'. Inaccurate. This is non-technical terminology and has no standing in engineering or physics.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            r"The definition of sensible heat ($\Delta T > 0$) perfectly mirrors the requirement of the question. The logic remains consistent with calorimetric principles where heat added equals the product of mass, specific heat, and temperature difference.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: B. sensible heat"
        )
    
    Example_7_Input = {
            "question": (
                "In a (direct expansion) evaporator, liquid refrigerant must boil away as close to the end of the coil as possible in order to:"
           ),
            "options": [
                "A. ensure proper oil return.",
                "B. ensure that frost does not accumulate.",
                "C. sub-cool the compressor.",
                "D. operate at high efficiency."
            ]
        }
    
    Example_7_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- System Topology: Direct Expansion (DX) Refrigeration System.\n"
            "- Component: Evaporator Coil.\n"
            "- Operating State: Refrigerant phase change (boiling/vaporization).\n"
            "- Constraint: Boiling must complete \"as close to the end of the coil as possible.\"\n"
            "- Goal: Determine the functional purpose of this specific heat transfer optimization.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "The efficiency of an evaporator is maximized when the internal surface area is utilized for latent heat transfer (boiling) rather than sensible heat transfer (superheating vapor). \n"
            "1. Latent Heat Transfer: High heat transfer coefficient; constant temperature.\n"
            "2. Superheating: Low heat transfer coefficient; provides protection for the compressor against liquid slugging.\n"
            "Completing the boil-off at the end of the coil ensures the maximum amount of evaporator surface area is used for high-capacity boiling while still achieving the necessary 5-10°F of superheat required for system safety.\n\n"

            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- 'Oil return' (Option A) is influenced by refrigerant velocity and piping slope, not primarily by the exact point where boiling finishes.\n"
            "- 'Frost accumulation' (Option B) is a function of surface temperature relative to the dew point and freezing point of air, not the internal phase-change endpoint.\n"
            "- 'Sub-cool' (Option C) is a term applied to the liquid line/condenser side; you cannot sub-cool a compressor using an evaporator (which produces superheated gas).\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'ensure proper oil return'. While gaseous refrigerant carries oil, the location of the boiling endpoint is not the primary engineering control for oil management.\n"
            "- Option B: 'ensure that frost does not accumulate'. Frost is managed by defrost cycles or maintaining saturated suction temperatures above freezing.\n"
            "- Option C: 'sub-cool the compressor'. This is a thermodynamic impossibility in this context; compressors are cooled by suction gas, but the process is not called 'sub-cooling'.\n"
            "- Option D: 'operate at high efficiency'. Physically valid. By utilizing almost the entire coil for boiling (latent heat), the system achieves the highest heat absorption rate per square foot of surface area.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "Maximizing the 'wetted' internal surface area of the evaporator while ensuring a dry, superheated gas reaches the compressor suction is the industry-standard definition of optimized DX evaporator performance.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: D. operate at high efficiency."
        )  
         
    
    Example_8_Input = {
            "question": (
                "What is the major difference between a heat pump and an air conditioner?"
           ),
            "options": [
                "A. Condenser",
                "B. Thermostatic expansion valve",
                "C. Evaporator",
                "D. Reversing valve"
            ]
        }
    
    Example_8_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- System Comparison: Air Conditioner (AC) vs. Heat Pump (HP).\n"
            "- Media Type: Refrigerant-based vapor compression cycle.\n"
            "- System Topology: Identification of the unique hardware component that enables multi-mode operation.\n"
            "- Operating State: Unidirectional cooling (AC) vs. Bidirectional heat transfer (Cooling/Heating in HP).\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "Both systems operate on the same refrigeration cycle (compression, condensation, expansion, and evaporation). The fundamental difference lies in the thermodynamic directionality. An air conditioner moves heat in one direction (indoor to outdoor), while a heat pump must be able to reverse the refrigerant flow to extract heat from the outdoors and reject it indoors during winter.\n\n"

            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- 'Condenser' and 'Evaporator' are 'False Friends' because while their *roles* swap in a heat pump, the physical heat exchangers exist in both systems.\n"
            "- The presence of an expansion valve (TXV) is common to both high-efficiency ACs and HPs, thus it is not the *major* differentiator for functional logic.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'Condenser'. Both systems require a condenser to reject heat; in a heat pump, the outdoor coil acts as the condenser in cooling and the evaporator in heating.\n"
            "- Option B: 'Thermostatic expansion valve'. A metering device used in both systems to regulate refrigerant flow; not unique to heat pumps.\n"
            "- Option C: 'Evaporator'. Both systems require an evaporator to absorb heat; it is a standard component of the vapor compression cycle.\n"
            "- Option D: 'Reversing valve'. Physically and functionally valid. This is a 4-way valve unique to heat pumps that redirects the high-pressure discharge gas to either the indoor or outdoor coil, enabling the change in mode.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The reasoning confirms that the reversing valve is the mechanical 'switch' that differentiates a cooling-only cycle from a reversible cycle. This aligns with the operational requirements of a heat pump.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: D. Reversing valve"
        )

    Example_9_Input = {
            "question": (
                "Which of the following is not a factor that should be considered when installing an outdoor unit?"
           ),
            "options": [
                "A. Return air",
                "B. Wind factors",
                "C. Sound transmission",
                "D. Snow fall"
            ]
        }
    
    Example_9_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- System Component: Outdoor unit (Condensing unit or Heat Pump outdoor section).\n"
            "- Activity: Installation and Siting.\n"
            "- Goal: Identify the parameter that is NOT a functional or regulatory constraint for the *outdoor* section of the system.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "The outdoor unit's primary thermodynamic function is heat rejection (in cooling mode) or heat absorption (in heating mode) via ambient air. This requires unrestricted airflow across the outdoor coil and consideration of environmental impacts on the hardware. \n"
            "- 'Return Air' is a specific term in the refrigeration cycle refers to the air being drawn from the conditioned *indoor* space to be reprocessed. It is a function of the indoor air handling system, not the outdoor heat rejection circuit.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- Distractor Check: While 'Air' is involved in both indoor and outdoor units, the 'Return Air' path is physically isolated from the outdoor unit's ambient air path. \n"
            "- Environmental constraints (Wind, Snow, Sound) are critical for outdoor unit longevity, performance, and local code compliance.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'Return air'. Physically invalid for outdoor units. This air stream belongs to the indoor ventilation loop. Its management does not affect the physical installation site or clearance of the outdoor condenser.\n"
            "- Option B: 'Wind factors'. Physically valid. Prevailing winds can cause fan motor 'windmilling' or impede heat rejection if blowing directly into the discharge air stream.\n"
            "- Option C: 'Sound transmission'. Physically valid. Local noise ordinances and proximity to bedroom windows or neighboring property lines dictate installation location.\n"
            "- Option D: 'Snow fall'. Physically valid. Outdoor units (especially heat pumps) must be elevated on 'snow legs' or stands to prevent the coil from being buried and to allow for proper drainage during defrost cycles.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The reasoning confirms that Options B, C, and D are standard site-selection criteria for exterior mechanical equipment. Option A is an indoor-side variable, making it the correct 'NOT' factor.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: A. Return air"   
        )
    
    Example_10_Input = {
            "question": (
                "One BTU is the amount of heat required to raise the temperature of:"
           ),
            "options": [
                "A. one pound of ice one-degree Fahrenheit.",
                "B. one pound of water one-degree Fahrenheit.",
                "C. one gallon of water one-degree Fahrenheit.",
                "D. one gallon of water eight degrees Fahrenheit."
            ]
        }
    
    Example_10_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- Unit of Measurement: British Thermal Unit (BTU).\n"
            "- Target Substance: Water (in a specific state).\n"
            "- Quantity: Standard mass or volume unit.\n"
            "- Thermodynamic Change: Sensible heat increase (temperature rise) of exactly one degree.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "The BTU is a traditional unit of heat defined by the heat capacity of liquid water. \n"
            r"The fundamental physical relationship is $Q = m \cdot c_p \cdot \Delta T$. \n"
            r"By definition, the specific heat ($c_p$) of liquid water is approximately $1.0 \text{ BTU/lb}\cdot^\circ\text{F}$. \n"
            r"Therefore, $1 \text{ BTU}$ is the energy required for $m = 1 \text{ lb}$ and $\Delta T = 1^\circ\text{F}$.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            r"- 'Ice' (Option A) is a distractor; the specific heat of ice is approximately $0.5 \text{ BTU/lb}\cdot^\circ\text{F}$, meaning it requires less energy than liquid water to raise its temperature.\n"
            r"- 'Gallon' (Options C and D) is a volume unit. Since a gallon of water weighs approximately $8.34 \text{ lbs}$, using a gallon would require $8.34 \text{ BTUs}$ for a $1^\circ\text{F}$ rise, violating the definition of a single BTU.\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'one pound of ice'. Incorrect. Specific heat of ice $\neq$ 1.0.\n"
            "- Option B: 'one pound of water one-degree Fahrenheit'. Correct. This perfectly matches the standard NIST and ASHRAE definition of the British Thermal Unit.\n"
            "- Option C: 'one gallon of water one-degree Fahrenheit'. Incorrect. Mass is too high ($~8.34 \text{ lbs}$).\n"
            "- Option D: 'one gallon of water eight degrees Fahrenheit'. Incorrect. This describes a much larger quantity of energy ($~66.7 \text{ BTUs}$).\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The reasoning confirms that the BTU is tied specifically to the mass of one pound of liquid water. This constant is the foundation for all IP (Imperial) unit HVAC calculations.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: B. one pound of water one-degree Fahrenheit."
        )
    
    Example_11_Input = {
            "question": ("A compressor is operating with a discharge pressure of 235.3 psig and a suction pressure of 35.3 psig. What is the compression ratio (pumping ratio)?"),
            "options": [
                "a. 10:1",
                "b. 8:1",
                "c. 7:1",
                "d. 5:1",
                
            ]}
    
    Example_11_output =(
        "Phase 1: Variable & Constraint Identification:\n"
        "Target Variable: Compression ratio (also known as pumping ratio).\n"
        "Known Parameters:\n"
        "- Discharge Pressure ($P_{d,\text{gauge}}$) = 235.3 psig\n"
        "- Suction Pressure ($P_{s,\text{gauge}}$) = 35.3 psig\n"
        "Implied Physical Constant: Standard atmospheric pressure = 14.7 psi.\n"
        "Constraint: The input values are in gauge pressure (psig), but thermodynamic ratios require absolute pressure (psia).\n"
        
        "Phase 2: Scientific/Engineering Principle Alignment\n"
        "HVAC Principle: The compression ratio of a compressor evaluates how much the compressor \"squeezes\" the refrigerant vapor. It is mathematically defined as the absolute discharge pressure divided by the absolute suction pressure ($CR = P_{d,\text{abs}} / P_{s,\text{abs}}$).\n"
        "Differential Check: The concept of an absolute pressure ratio must be contrasted with a gauge pressure ratio. Dividing the given gauge pressures directly is a common trap and will result in a functionally useless metric, as it ignores the ambient atmospheric pressure already acting on the system.\n"
        
        "Phase 3: Boundary Condition & 'Industry Context' Check\n"
        "Technical Level: Vocational/Field Technician level. This tests a fundamental mathematical step used to diagnose compressor efficiency and health.\n"
        "Distractors & Constants: The core test here is the industry-standard conversion formula: PSIG + 14.7 = PSIA. In field contexts (like ASHRAE guidelines or EPA tests), standard atmospheric pressure at sea level (14.7) is used universally unless altitude is specified. Therefore, adjusting the field gauge readings to absolute physical truth is the mandatory first boundary condition.\n"
        
        "Phase 4: Independent Option Evaluation\n"
        "Step 1: Convert to Absolute Pressure (PSIA)\n"
        "- Absolute Discharge Pressure = 235.3 + 14.7 = 250 psia\n"
        "- Absolute Suction Pressure = 35.3 + 14.7 = 50 psia\n"
        "Step 2: Calculate the Ratio\n"
        "- Ratio = 250 / 50 = 5\n"
        "Option A (10:1): Incorrect. Does not map to any logical miscalculation of the provided variables.\n"
        "Option B (8:1): Incorrect. Mathematically invalid.\n"
        "Option C (7:1): Incorrect. This is a distractor likely aimed at technicians who fail to add 14.7 and mistakenly divide the gauge pressures directly ($235.3 / 35.3 \\approx 6.66$, rounded to 7).\n"
        "Option D (5:1): Correct. The mathematically precise result of $250 \\text{ psia} / 50 \\text{ psia}$.\n"
        
        "Phase 5: Internal Consistency Check\n"
        "The 'Indicator over Category' Rule: A technician physically reads the gauge pressure (PSIG) off their manifold gauges in the field. However, to understand the thermodynamic work the compressor is actually doing (the category), the indicator must be converted to absolute terms.\n"
        "Logical Loop-back: Does the answer make sense? Yes, a 5:1 compression ratio is highly typical and safe for a standard reciprocating or scroll compressor in medium-temp refrigeration or standard air conditioning applications.\n"
        
        "Phase 6: Final Conclusion\n"
        "Option d (5:1) is the correct answer."
    )

    Example_12_Input = {
            "question": ("How many amps can a 40VA, 120-volt transformer with a 24-volt secondary carry?"),
            "options": [
                "A. 2.0",
                "B. 1.67",
                "C. 3.0",
                "D. 0.3"
            ]
        }

    Example_12_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "Extract every technical parameter:\n"
            "Apparent Power (Capacity): S = 40 VA\n"
            "Primary Voltage (Line Voltage): Vp = 120 V\n"
            "Secondary Voltage (Control Voltage): Vs = 24 V\n"
            "Physical constraint: The question asks how many amps the transformer can 'carry'. In electrical terminology, 'carry' refers to the load it can supply, which strictly points to the secondary (output) circuit capacity.\n"

            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "Identify the specific HVAC principle:** The fundamental principle is the AC Power Formula: Apparent Power (VA) = Voltage (V) * Current (I). Therefore, I = VA / V.\n"
            "Differential Check:** Primary side vs. Secondary side.\n"
            "Primary current (Input draw): Ip = S / Vp\n"
            "Secondary current (Output load capacity): Is = S / Vs\n"

            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "Technical level:** Vocational HVAC electrical troubleshooting / NATE Certification level.\n" 
            "Industry Context (ASHRAE/Field Terminology):** A 120V to 24V 40VA transformer is the absolute industry standard for residential HVAC control circuits. When field technicians ask 'what can it carry,' they are always sizing the low-voltage (24V) control circuit.\n"
            "Distractor Check:** If a candidate incorrectly uses the primary voltage (120V), the calculation is 40 / 120 = 0.33 A. Option D (0.3) is deliberately placed as a distractor.\n"

            "Phase 4: Independent Option Evaluation:\n"
            "Option A (2.0):** Mathematically invalid for both primary and secondary circuits. (40 / 24 != 2.0)\n"
            "Option B (1.67):** Calculated as secondary capacity: Is = 40 VA / 24 V = 1.666... A. Physically and mathematically valid.\n"
            "Option C (3.0):** Incorrect. Field context distractor. 3.0 Amps is the typical rating for the automotive blade fuse used to protect the circuit, not the capacity itself.\n"
            "Option D (0.3):** Incorrect. Calculated as 40 / 120 = 0.33 A. This is the primary side current draw.\n"

            "Phase 5: Internal Consistency Check:\n"
            "The 'Indicator over Category' Rule:** In HVAC field diagnostics, a technician uses a multimeter (Amp Clamp) on the 24V secondary wire. The parameter they are measuring against is 1.67 A.\n" 
            "Logical Loop-back:** Does 1.67 A explain how a technician distinguishes a healthy state from a fault? Yes. If the ammeter reads >1.67 A, the technician knows a component is drawing excessive current.\n"

            "Phase 6: Final Conclusion:\n"
            "Based strictly on the verified AC power formula and HVAC field application logic for secondary load carrying capacity, Option B is the only mathematically and contextually correct answer.\n"

            "Correct Answer: b 1.67\n"
    )

    Example_13_Input = {
            "question": ("Calculate the amount of heat required to change 5 lbs of 32°F ice to water at 32°F."),
            "options": [
                "A. No extra heat is required because the ice is already at the melting temperature.",
                "B. 5 lbs x 144 BTUs/lb = 720 BTUs",
                "C. 5 lbs x 32oF x 0.5 BTU/lb = 80 BTUs",
                "D. The amount of heat required cannot be calculated without knowing the ending temperature of the water."
            ]
        }


    Example_13_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "Extract every technical parameter:\n" 
            "Mass (m) = 5 lbs.\n"
            "Initial State = Ice (Solid) at 32°F.\n"
            "Final State = Water (Liquid).\n"
            "Physical constraint: The calculation must strictly account for the change of state (melting process). Unless an ending temperature above 32°F is explicitly specified, industry standard problems assume the calculation ends exactly when the state change is complete (32°F water).\n"

            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "Identify the specific HVAC principle: Latent Heat of Fusion. The standard HVAC/ASHRAE constant for melting ice is 144 BTU/lb. The formula is Q = m * L_f.\n"
            "Differential Check: \n"
            "Latent Heat: Requires energy to break molecular bonds without changing the temperature.\n"
            "Contrast: Sensible Heat: Requires energy to change temperature without changing the state.\n"

            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "Technical level: Vocational HVAC Thermodynamics / NATE Core Fundamentals.\n"
            "Industry Context (ASHRAE/Field Terminology): 144 BTU/lb is a foundational constant in HVAC. \n"
            "Distractor Check: Options testing common misconceptions (e.g., confusing temperature with heat energy, or misapplying sensible heat formulas to latent heat processes) are typical distractors in certification exams.\n"

            "Phase 4: Independent Option Evaluation:\n"
            "Option A (No extra heat...): Incorrect. Conceptual distractor. It confuses temperature with heat energy. Even though the ice is at the melting point, a massive amount of 'hidden' (latent) heat must be absorbed to break the solid molecular bonds into a liquid state.\n"
            "Option B (5 lbs x 144 BTUs/lb = 720 BTUs): Correct. Mathematically and physically valid. It correctly uses the mass (5 lbs) and the standard latent heat of fusion constant for water (144 BTU/lb). \n"
            "Option C (5 lbs x 32°F x 0.5 BTU/lb = 80 BTUs): Incorrect. Mathematical distractor. This incorrectly applies the sensible heat formula (Q = m * c * ΔT), uses the specific heat of ice (0.5), and falsely uses the absolute temperature (32) as if it were a temperature change (ΔT).\n"
            "Option D (Cannot be calculated without ending temperature...): Incorrect. In standard HVAC thermodynamics questions, changing 'ice to water' implies calculating the phase change exactly at the melting point.\n"

            "Phase 5: Internal Consistency Check:\n"
            "The 'Indicator over Category' Rule: If a technician places a thermometer in the container, it will read exactly 32°F the entire time the 720 BTUs are being absorbed. The thermometer does not move, which perfectly invalidates Option A and Option D.\n"
            "Logical Loop-back: Does Option B completely explain the state change? Yes, 720 BTUs completely satisfies the physical requirement to transition 5 lbs of solid H2O to liquid H2O.\n"

            "Phase 6: Final Conclusion:\n"
            "Based strictly on the principles of latent heat of fusion and standard HVAC exam assumptions for phase changes, Option B is the only mathematically, physically, and conceptually correct answer.\n"

            "Correct Answer: b) 5 lbs x 144 BTUs/lb = 720 BTUs\n"
)

    Example_14_Input = {
            "question": (
                "Which of the following is an example of a resistive load?"
           ),
            "options": [
                "A. Bimetal switch",
                "B. Crankcase heater",
                "C. Transformer",
                "D. Motor"
            ]
        }
    
    Example_14_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- Target Classification: Resistive Load.\n"
            r"- Load Characteristic: Energy conversion primarily into heat; Voltage and Current are in phase ($\phi = 0$); Power Factor ($PF$) = 1.0.\n"
            "- System Context: Electrical components within an HVAC or mechanical system.\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            "In AC circuit theory, loads are classified by their impedance ($Z$). \n"
            "1. Resistive Loads ($R$): Dissipate energy via Joule heating ($P = I^2R$).\n"
            "2. Inductive Loads ($L$): Store energy in a magnetic field, causing current to lag voltage (e.g., coils, motors).\n"
            "3. Capacitive Loads ($C$): Store energy in an electric field, causing current to lead voltage.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- 'Bimetal switch' (Option A) is a control device (conductor/interrupter), not a load. While it has negligible resistance, its functional purpose is switching, not energy consumption.\n"
            "- 'Transformer' and 'Motor' (Options C and D) are common inductive loads due to their reliance on electromagnetic induction (windings).\n\n"
    
            "Phase 4: Independent Option Evaluation:\n"
            "- Option A: 'Bimetal switch'. Physically, this is a thermally actuated contact. It is part of the control circuit logic, not a load that consumes power to perform work or generate heat.\n"
            "- Option B: 'Crankcase heater'. Physically valid. This is a pure resistance heating element designed to keep compressor oil warm. It converts electrical energy directly into thermal energy with no significant inductive reactance.\n"
            "- Option C: 'Transformer'. Physically invalid as a resistive load. It operates on mutual induction, making it an inductive load characterized by magnetic flux and windings.\n"
            "- Option D: 'Motor'. Physically invalid as a resistive load. It utilizes copper windings to create torque via magnetism, making it a highly inductive load.\n\n"
    
            "Phase 5: Internal Consistency Check:\n"
            "The reasoning identifies that a crankcase heater is a resistive heating element, which aligns with the physical definition of a resistive load (conversion of electricity to heat). This distinguishes it clearly from the inductive nature of motors and transformers.\n\n"
    
            "Phase 6: Final Conclusion:\n"
            "The correct answer is: B. Crankcase heater"
        )

    Example_15_Input = {
            "question": (
                "When the temperatures of a structure both inside and outside are equal, there is _____."
           ),
            "options": [
                "A. no heat transfer",
                "B. latent heat transfer to the outside",
                "C. thermal heat transfer of sensible heat",
                "D. a lower rate of relative humidity"
            ]
        }
    
    Example_15_output =(
            "Phase 1: Variable & Constraint Identification:\n"
            "- System Topology: A structure (thermal envelope).\n"
            "- Operating State: Thermal equilibrium.\n"
            "- Primary Constraint: Inside temperature ($T_{in}$) is equal to outside temperature ($T_{out}$).\n"
            r"- Goal: Determine the resultant heat transfer state based on temperature differential ($\Delta T$).\n\n"
    
            "Phase 2: Scientific/Engineering Principle Alignment:\n"
            r"The Second Law of Thermodynamics and Fourier's Law of Heat Conduction ($Q = k \cdot A \cdot \frac{\Delta T}{d}$) dictate that a temperature gradient is the fundamental driving force for heat transfer. Heat naturally flows from a higher temperature region to a lower temperature region. If $\Delta T = 0$, the driving potential for heat transfer is eliminated.\n\n"
    
            "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
            "- 'Latent heat transfer' (Option B) requires a vapor pressure or moisture content differential, which is not defined here. \n"
            "- 'Sensible heat transfer' (Option C) specifically requires a temperature difference by definition.\n"
            "- 'Relative humidity' (Option D) is a ratio of vapor pressure and is not directly dictated solely by the equality of dry-bulb temperatures without knowing moisture levels.\n\n"
    
           "Phase 4: Independent Option Evaluation:\n"
           r"- Option A: 'no heat transfer'. Physically valid. With no temperature gradient ($\Delta T = 0$), there is no net energy exchange between the two environments.\n"
           "- Option B: 'latent heat transfer to the outside'. Physically invalid. There is no evidence of a humidity gradient, and latent heat typically requires a phase change process not implied by simple temperature equality.\n"
           "- Option C: 'thermal heat transfer of sensible heat'. Physically invalid. Sensible heat transfer is the product of mass, specific heat, and a temperature change; since temperatures are equal, this value is zero.\n"
           "- Option D: 'a lower rate of relative humidity'. Inaccurate. Temperature equality does not inherently result in a lower relative humidity; RH depends on the specific moisture content (mass of water vapor) in each air mass.\n\n"
    
           "Phase 5: Internal Consistency Check:\n"
           "The reasoning confirms that heat transfer is a function of potential difference. Zero potential difference results in zero transfer. This logic remains internally consistent across all thermodynamic laws (conduction, convection, and radiation).\n\n"
    
           "Phase 6: Final Conclusion:\n"
           "The correct answer is: A. no heat transfer"
        )
    
    Example_16_Input = {
            "question": (
                "R-407C has _______."
           ),
            "options": [
                "A. a foul odor",
                "B. to be charged in the vapor phase",
                "C. the ability to fractionate",
                "D. no temperature glide"
            ]
        }
    
    Example_16_output =(
           "Phase 1: Variable & Constraint Identification:\n"
           "- Media Type: R-407C refrigerant.\n"
           "- Chemical Composition: Ternary Zeotropic Blend (R-32 / R-125 / R-134a in a 23/25/52 ratio).\n"
           "- Operating State: Phase change (evaporation/condensation) within a refrigeration cycle.\n"
           "- Goal: Identify the physical behavior characteristic of this specific refrigerant blend.\n\n"
    
           "Phase 2: Scientific/Engineering Principle Alignment:\n"
           "R-407C belongs to the 400-series refrigerants, which are zeotropic blends. \n"
           "1. Dalton's Law & Raoult's Law: Because the constituent refrigerants have different boiling points and vapor pressures, they do not evaporate or condense at a single temperature for a given pressure.\n"
           "2. Fractionation: This refers to the change in composition of the refrigerant blend when it leaks or changes phase, as the component with the highest vapor pressure escapes or boils off faster than the others.\n"
           "3. Temperature Glide: The difference between the saturated vapor temperature (dew point) and the saturated liquid temperature (bubble point).\n\n"

           "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
           r"- 'No temperature glide' (Option D) is a 'false friend'; this only applies to azeotropic blends (500-series) or pure refrigerants. R-407C has a significant glide (approx. $10-12^\circ\text{F}$).\n"
           "- 'Vapor phase charging' (Option B) is a dangerous practice for blends; fractionation during charging would cause the system to have the wrong chemical ratio. Zeotropic blends must always be liquid-charged.\n\n"
    
           "Phase 4: Independent Option Evaluation:\n"
           "- Option A: 'a foul odor'. Physically invalid. Like most HFCs, R-407C is practically odorless and colorless in its pure state.\n"
           "- Option B: 'to be charged in the vapor phase'. Physically invalid. Charging as a vapor would lead to fractionation. Blends must be removed from the cylinder as a liquid to maintain the correct component ratio.\n"
           "- Option C: 'the ability to fractionate'. Physically valid. Due to the different boiling points of its components (R-32, R-125, R-134a), the blend will separate/fractionate if allowed to change phase in a non-static environment (like a slow leak).\n"
           "- Option D: 'no temperature glide'. Physically invalid. As a zeotropic blend, R-407C inherently exhibits temperature glide during phase change.\n\n"
    
           "Phase 5: Internal Consistency Check:\n"
           "The reasoning confirms that R-407C is a non-azeotropic (zeotropic) mixture. The physical consequence of different vapor pressures in a single mixture is the tendency to fractionate. This aligns perfectly with Option C.\n\n"
    
           "Phase 6: Final Conclusion:\n"
           "The correct answer is: C. the ability to fractionate" 
        )
    
    Example_17_Input = {
            "question": (
                "What is a carbon footprint?"
           ),
            "options": [
                "A. The carbon deposits from burning gasoline.",
                "B. The amount of carbon dioxide that is produced to support your lifestyle.",
                "C. The amount of carbon in the atmosphere produced by the world's lifestyle.",
                "D. The amount of carbon in the stratosphere."
            ]
        }
    
    Example_17_output =(
           "Phase 1: Variable & Constraint Identification:\n"
           "- Target Concept: Carbon Footprint.\n"
           "- Scope: Individual vs. Global vs. Environmental deposits.\n"
           "- Media Type: Greenhouse gas emissions (primarily $CO_2$ and equivalents).\n"
           "- Operating State: Ongoing lifecycle activities (transportation, energy use, consumption).\n\n"
    
           "Phase 2: Scientific/Engineering Principle Alignment:\n"
           "The carbon footprint is a functional metric used in environmental engineering and life cycle assessment (LCA). It quantifies the total greenhouse gas (GHG) emissions caused directly and indirectly by an individual, organization, event, or product. It is expressed as carbon dioxide equivalent ($CO_2e$) to standardize the global warming potential (GWP) of various gases produced by human activity.\n\n"
    
           "Phase 3: Boundary Condition & 'Industry Context' Check:\n"
           "- 'Carbon deposits' (Option A) is a 'false friend'; it refers to physical soot or residue, not the atmospheric emission metric.\n"
           "- 'Stratosphere' (Option D) is a boundary error; most human-produced carbon impact occurs in the troposphere and is measured as a total atmospheric contribution, not isolated to the stratosphere.\n"
           "- The distinction between 'individual lifestyle' (Option B) and 'world's lifestyle' (Option C) is critical; while both are footprints, the standard definition of 'a' carbon footprint usually refers to the specific entity (person/product) in question.\n\n"
    
           "Phase 4: Independent Option Evaluation:\n"
           "- Option A: 'The carbon deposits from burning gasoline'. Physically invalid. These are localized particulate matters (soot), not the comprehensive gaseous emission metric defined as a footprint.\n"
           "- Option B: 'The amount of carbon dioxide that is produced to support your lifestyle'. Physically valid. This correctly identifies the lifecycle emissions tied to individual activities and energy consumption.\n"
           "- Option C: 'The amount of carbon in the atmosphere produced by the world's lifestyle'. Technically this describes the *global* carbon footprint or total anthropogenic emissions, but the term 'a carbon footprint' typically scales to the specific subject (you/your).\n"
           "- Option D: 'The amount of carbon in the stratosphere'. Physically invalid. $CO_2$ is well-mixed in the atmosphere, but the term 'footprint' describes an emission source/total, not a concentration in a specific atmospheric layer.\n\n"
    
           "Phase 5: Internal Consistency Check:\n"
           "The reasoning identifies that a 'footprint' is a measure of impact relative to an agent's activities. Defining it as the $CO_2$ produced to support an individual lifestyle is internally consistent with environmental accounting standards.\n\n"
    
           "Phase 6: Final Conclusion:\n"
           "The correct answer is: B. The amount of carbon dioxide that is produced to support your lifestyle."
        )
    
    Example_18_Input = {
            "question": (
                "The safety ground conductor for A/C circuit is usually color coded _________."
           ),
            "options": [
                "A. red",
                "B. green",
                "C. black",
                "D. white"
            ]
        }
    
    Example_18_output =(
           "Phase 1: Variable & Constraint Identification:\n"
           "- Media Type: Electrical conductor (A/C circuit).\n"
           "- Component: Safety ground conductor (Equipment Grounding Conductor - EGC).\n"
           "- Regulatory Anchor: NEC (National Electrical Code) / NFPA 70.\n"
           "- Goal: Identify the standardized color-coding for safety identification.\n\n"
    
           "Phase 2: Scientific/Engineering Principle Alignment:\n"
           r"Electrical safety protocols require a non-current-carrying path to earth to prevent electric shock and fire. \n"
           r"- Differential Check: Unlike 'Neutral' (grounded conductor) or 'Hot' (ungrounded conductor), the 'Safety Ground' is dedicated to fault current. \n"
           r"- Unique Physical Marker: Color-coding is the primary visual 'marker' used in field diagnostics to prevent hazardous cross-wiring between current-carrying and safety-path conductors.\n\n"
    
           r"Phase 3: Boundary Condition & 'Industry Context' Check:\n"
           r"- Technical Level: Vocational/Introductory electrical standards.\n"
           r"- Distractor Check: White (Neutral) is often confused with Ground by novices because both connect to the ground bus in a main panel, but they serve different functional roles in the circuit topology.\n\n"
    
           r"Phase 4: Independent Option Evaluation:\n"
           r"- Option A: 'red'. Physically invalid for grounding. Typically used for 'Hot' legs in 240V circuits or secondary switch legs.\n"
           r"- Option B: 'green'. Physically and legally valid. Per NEC Article 250, grounding conductors must be green, green with yellow stripes, or bare copper.\n"
           r"- Option C: 'black'. Physically invalid for grounding. Standard code for 'Hot' (L1) ungrounded conductors in North American AC systems.\n"
           r"- Option D: 'white'. Physically invalid for grounding. This color is reserved for the 'Neutral' (grounded) current-carrying conductor.\n\n"

           r"Phase 5: Internal Consistency Check:\n"
           r"- Indicator over Category: In field diagnostics, a technician distinguishes the safety path from the return path strictly by color-coding (Green vs. White). \n"
           r"- Logical Loop-back: Selecting 'Green' ensures that even in a fault condition, the safety path remains distinguishable from the neutral, preventing accidental energization of equipment chassis.\n\n"
    
           r"Phase 6: Final Conclusion:\n"
           r"The correct answer is: B. green"
        )
    # 5. API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": CUSTOM_INSTRUCTION},
            {"role": "user", "content": json.dumps(Example_1_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_1_output},
            {"role": "user", "content": json.dumps(Example_2_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_2_output},
            {"role": "user", "content": json.dumps(Example_3_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_3_output},
            {"role": "user", "content": json.dumps(Example_4_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_4_output},
            {"role": "user", "content": json.dumps(Example_5_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_5_output},
            {"role": "user", "content": json.dumps(Example_6_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_6_output},
            {"role": "user", "content": json.dumps(Example_7_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_7_output},
            {"role": "user", "content": json.dumps(Example_8_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_8_output},
            {"role": "user", "content": json.dumps(Example_9_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_9_output},
            {"role": "user", "content": json.dumps(Example_10_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_10_output},
            {"role": "user", "content": json.dumps(Example_11_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_11_output},
            {"role": "user", "content": json.dumps(Example_12_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_12_output},
            {"role": "user", "content": json.dumps(Example_13_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_13_output},
            {"role": "user", "content": json.dumps(Example_17_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_17_output},
            {"role": "user", "content": json.dumps(Example_18_Input, ensure_ascii=False)},
            {"role": "assistant", "content": Example_18_output},
            {"role": "user", "content": json.dumps(Structured_input, ensure_ascii=False)}
        ],
        temperature=0,
    )
    
    return response.choices[0].message.content



if __name__ == "__main__":
    
    raw_input = input("question: ") 

    try:
        
        structured_data = input_processor(raw_input)
        print(f"✅ : {structured_data}")

        
        final_report = get_custom_response(structured_data)
        print(final_report)
        
    except Exception as e:
        print(f"❌ : {e}")


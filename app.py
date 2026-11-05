# ============================================================
# SEMICONDUCTOR ELECTRONIC PROPERTY PREDICTION SYSTEM
# Streamlit Application
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import accuracy_score


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="2D Semiconductor Prediction System",

    page_icon="🔬",

    layout="wide",

    initial_sidebar_state="expanded"

)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================
#
#BASE_DIR = os.path.dirname(
#    os.path.abspath(__file__)
#)

#MODEL_PATH = os.path.join(
#    BASE_DIR,
#    "Results",
#    "Models",
#    "Linear_SVM_complete_pipeline.joblib"
#)

#CONFIG_PATH = os.path.join(
#    BASE_DIR,
#    "Results",
#    "Models",
#    "deployment_config.json"
#)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Linear_SVM_complete_pipeline.joblib"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "deployment_config.json"
)

# ============================================================
# 3. LOAD MODEL AND CONFIGURATION
# ============================================================


@st.cache_resource(show_spinner="Loading semiconductor classification model...")
def load_model():

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():

    with open(CONFIG_PATH, "r") as file:

        return json.load(file)


try:

    model = load_model()

    config = load_config()

except Exception as error:

    st.error(
        f"Unable to load deployment resources: {error}"
    )

    st.stop()


# ============================================================
# 4. APPLICATION METADATA
# ============================================================

PROJECT_NAME = config["project_name"]

TARGET_CLASSES = config["target_classes"]

FEATURE_COLUMNS = config["feature_columns"]

NUMERICAL_FEATURES = config["numerical_features"]

CATEGORICAL_FEATURES = config["categorical_features"]

MATERIAL_NAMES = config["material_names"]

SPACE_GROUPS = config["space_groups"]

NUMERICAL_DEFAULTS = config["numerical_defaults"]


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown("""

<style>

.main-header {

    font-size: 32px;

    font-weight: 700;

    color: #1F4E79;

    margin-bottom: 5px;

}

.sub-header {

    font-size: 16px;

    color: #666666;

    margin-bottom: 25px;

}

.metric-card {

    background-color: #F4F7FB;

    padding: 20px;

    border-radius: 10px;

    text-align: center;

}

.result-header {

    font-size: 24px;

    font-weight: bold;

    color: #1F4E79;

}

</style>

""", unsafe_allow_html=True)


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("SEMICONDUCTOR AI")

st.sidebar.caption(
    "2D Electronic Property Classification"
)

page = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Electronic Property Prediction",

        "Batch Prediction",

        "Model Information"

    ]

)

st.sidebar.divider()

st.sidebar.write(
    f"Selected Model: **{config['selected_model']}**"
)


# ============================================================
# 7. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-header">'
        '2D Semiconductor Electronic Property Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-header">'
        'Machine learning-based classification of electronic '
        'properties in two-dimensional semiconductor materials.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Input Features",
        len(FEATURE_COLUMNS)
    )

    col2.metric(
        "Target Classes",
        len(TARGET_CLASSES)
    )

    col3.metric(
        "Materials",
        len(MATERIAL_NAMES)
    )

    col4.metric(
        "Model",
        config["selected_model"].replace("_", " ")
    )

    st.divider()

    st.subheader("Electronic Property Categories")

    class_data = pd.DataFrame({

        "Electronic Property": TARGET_CLASSES,

        "Description": [

            "Band Gap",

            "Valence Band Maximum",

            "Conduction Band Minimum"

        ] if set(TARGET_CLASSES) == {
            "Band Gap", "VBM", "CBM"
        } else TARGET_CLASSES

    })

    st.dataframe(
        class_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("System Overview")

    st.write("""

    This application uses a trained machine learning pipeline
    to classify the electronic-property category associated
    with supplied semiconductor descriptors.

    Users can submit individual material descriptors or upload
    a CSV file containing multiple observations.

    The model automatically performs numerical scaling,
    categorical encoding and classification.

    """)


# ============================================================
# 8. SINGLE ELECTRONIC PROPERTY PREDICTION
# ============================================================

elif page == "Electronic Property Prediction":

    st.title("Electronic Property Prediction")

    st.write(
        "Enter the material descriptors to obtain a prediction."
    )

    with st.form("prediction_form"):

        st.subheader("A. Material Information")

        col1, col2 = st.columns(2)

        with col1:

            material_name = st.selectbox(

                "Material Name",

                MATERIAL_NAMES

            )

        with col2:

            space_group = st.selectbox(

                "Crystal Symmetry Space Group",

                SPACE_GROUPS

            )

        st.divider()

        st.subheader("B. Atomic Descriptors")

        col1, col2, col3 = st.columns(3)

        with col1:

            atomic_number = st.number_input(

                "Atomic Number",

                min_value=1,

                max_value=118,

                value=39

            )

            atomic_radius = st.number_input(

                "Atomic Radius (pm)",

                min_value=0.0,

                value=float(
                    NUMERICAL_DEFAULTS["Atomic_Radius_pm"]
                ),

                step=0.1

            )

        with col2:

            electronegativity = st.number_input(

                "Electronegativity",

                min_value=0.0,

                value=float(
                    NUMERICAL_DEFAULTS["Electronegativity"]
                ),

                step=0.01

            )

            valence_electrons = st.number_input(

                "Valence Electrons",

                min_value=0,

                max_value=20,

                value=4

            )

        with col3:

            ionization_energy = st.number_input(

                "Ionization Energy (eV)",

                min_value=0.0,

                value=float(
                    NUMERICAL_DEFAULTS["Ionization_Energy_eV"]
                ),

                step=0.01

            )

            coordination_number = st.number_input(

                "Coordination Number",

                min_value=0,

                max_value=20,

                value=5

            )

        st.divider()

        st.subheader("C. Structural Descriptors")

        col1, col2, col3 = st.columns(3)

        with col1:

            lattice_a = st.number_input(

                "Lattice Parameter a (Å)",

                min_value=0.0,

                value=float(
                    NUMERICAL_DEFAULTS["Lattice_Parameter_a_A"]
                ),

                step=0.01

            )

        with col2:

            lattice_b = st.number_input(

                "Lattice Parameter b (Å)",

                min_value=0.0,

                value=float(
                    NUMERICAL_DEFAULTS["Lattice_Parameter_b_A"]
                ),

                step=0.01

            )

        with col3:

            bond_length = st.number_input(

                "Average Bond Length (Å)",

                min_value=0.0,

                value=float(
                    NUMERICAL_DEFAULTS["Average_Bond_Length_A"]
                ),

                step=0.01

            )

        layer_thickness = st.number_input(

            "Layer Thickness (Å)",

            min_value=0.0,

            value=float(
                NUMERICAL_DEFAULTS["Layer_Thickness_A"]
            ),

            step=0.01

        )

        st.divider()

        st.subheader("D. Electronic Descriptors")

        col1, col2, col3 = st.columns(3)

        with col1:

            formation_energy = st.number_input(

                "Formation Energy (eV/atom)",

                value=float(
                    NUMERICAL_DEFAULTS["Formation_Energy_eV_atom"]
                ),

                step=0.01

            )

        with col2:

            fermi_level = st.number_input(

                "Fermi Level (eV)",

                value=float(
                    NUMERICAL_DEFAULTS["Fermi_Level_eV"]
                ),

                step=0.01

            )

        with col3:

            work_function = st.number_input(

                "Work Function (eV)",

                min_value=0.0,

                value=float(
                    NUMERICAL_DEFAULTS["Work_Function_eV"]
                ),

                step=0.01

            )

        col1, col2 = st.columns(2)

        with col1:

            dos_peak = st.number_input(

                "DOS Peak Value",

                value=float(
                    NUMERICAL_DEFAULTS["DOS_Peak_Value"]
                ),

                step=0.01

            )

        with col2:

            st.info(
                "All 16 original predictor variables "
                "are required by the trained pipeline."
            )

        submitted = st.form_submit_button(

            "Predict Electronic Property",

            use_container_width=True

        )


    # ========================================================
    # 9. GENERATE PREDICTION
    # ========================================================

    if submitted:

        input_data = pd.DataFrame({

            "Material_Name": [material_name],

            "Atomic_Number": [atomic_number],

            "Electronegativity": [electronegativity],

            "Atomic_Radius_pm": [atomic_radius],

            "Valence_Electrons": [valence_electrons],

            "Ionization_Energy_eV": [ionization_energy],

            "Lattice_Parameter_a_A": [lattice_a],

            "Lattice_Parameter_b_A": [lattice_b],

            "Crystal_Symmetry_Space_Group": [space_group],

            "Coordination_Number": [coordination_number],

            "Formation_Energy_eV_atom": [formation_energy],

            "Fermi_Level_eV": [fermi_level],

            "Work_Function_eV": [work_function],

            "DOS_Peak_Value": [dos_peak],

            "Average_Bond_Length_A": [bond_length],

            "Layer_Thickness_A": [layer_thickness]

        })

        input_data = input_data[
            FEATURE_COLUMNS
        ]

        prediction = model.predict(
            input_data
        )[0]

        probabilities = model.predict_proba(
            input_data
        )[0]

        st.divider()

        st.subheader("Prediction Results")

        st.markdown(

            f"""
            <div class="result-header">
            Predicted Electronic Property: {prediction}
            </div>
            """,

            unsafe_allow_html=True

        )

        st.metric(

            "Model Confidence",

            f"{probabilities.max() * 100:.2f}%"

        )

        probability_df = pd.DataFrame({

            "Electronic Property": model.classes_,

            "Probability": probabilities

        })

        fig = px.bar(

            probability_df,

            x="Electronic Property",

            y="Probability",

            color="Electronic Property",

            title="Class Prediction Probabilities",

            text=probability_df["Probability"].map(
                lambda x: f"{x:.3f}"
            )

        )

        fig.update_layout(
            yaxis_range=[0, 1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Input Descriptors")

        st.dataframe(
            input_data,
            use_container_width=True
        )

        st.download_button(

            "Download Prediction",

            input_data.assign(

                Predicted_EPV=prediction,

                Prediction_Confidence=probabilities.max()

            ).to_csv(index=False),

            file_name="semiconductor_prediction.csv",

            mime="text/csv"

        )


# ============================================================
# 10. BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.title("Batch Electronic Property Prediction")

    st.write("""

    Upload a CSV file containing the 16 original predictors.
    The application will classify all submitted observations.

    """)

    uploaded_file = st.file_uploader(

        "Upload Semiconductor Dataset",

        type=["csv"]

    )

    if uploaded_file is not None:

        batch_data = pd.read_csv(
            uploaded_file
        )

        st.subheader("Uploaded Dataset")

        st.write(
            f"Observations: {len(batch_data):,}"
        )

        st.dataframe(
            batch_data.head(),
            use_container_width=True
        )

        missing_columns = set(
            FEATURE_COLUMNS
        ) - set(batch_data.columns)

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

        else:

            if st.button(
                "Run Batch Prediction",
                use_container_width=True
            ):

                prediction_input = batch_data[
                    FEATURE_COLUMNS
                ].copy()

                predictions = model.predict(
                    prediction_input
                )

                probabilities = model.predict_proba(
                    prediction_input
                )

                batch_data["Predicted_EPV"] = (
                    predictions
                )

                batch_data["Prediction_Confidence"] = (
                    probabilities.max(axis=1)
                )

                for index, class_name in enumerate(
                    model.classes_
                ):

                    batch_data[
                        f"Probability_{class_name.replace(' ', '_')}"
                    ] = probabilities[:, index]

                st.success(
                    "Batch prediction completed."
                )

                st.subheader("Prediction Results")

                st.dataframe(
                    batch_data,
                    use_container_width=True
                )

                st.download_button(

                    "Download Results",

                    batch_data.to_csv(index=False),

                    file_name="semiconductor_batch_predictions.csv",

                    mime="text/csv"

                )


# ============================================================
# 11. MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.title("Model Information")

    st.subheader("Deployment Configuration")

    model_information = {

        "Project": PROJECT_NAME,

        "Selected Model": config["selected_model"],

        "Number of Predictors": len(FEATURE_COLUMNS),

        "Numerical Features": len(NUMERICAL_FEATURES),

        "Categorical Features": len(CATEGORICAL_FEATURES),

        "Target Variable": config["target"],

        "Target Classes": ", ".join(TARGET_CLASSES),

        "Model File": os.path.basename(MODEL_PATH)

    }

    information_df = pd.DataFrame({

        "Parameter": model_information.keys(),

        "Value": model_information.values()

    })

    st.dataframe(
        information_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Input Features")

    st.write("Numerical Features")

    st.write(NUMERICAL_FEATURES)

    st.write("Categorical Features")

    st.write(CATEGORICAL_FEATURES)

    st.subheader("Target Classes")

    st.write(TARGET_CLASSES)

    st.warning("""

    The model provides classification estimates based on the
    supplied dataset and its learned statistical relationships.

    These predictions should not be interpreted as experimentally
    validated electronic properties of semiconductor materials.

    """)


# ============================================================
# 12. APPLICATION FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "2D Semiconductor Electronic Property Classification"
)

st.sidebar.caption(
    "Research Prototype | Machine Learning Deployment"
)
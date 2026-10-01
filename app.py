
import streamlit as st
import json
import os
import base64
import io

from PIL import Image
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Recipe & Pantry Helper",
    layout="wide"
)


# =========================================================
# CUSTOM THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #FCE4DF !important;
    color: #3D2929 !important;
}

.main {
    background-color: #FCE4DF !important;
}

[data-testid="stHeader"] {
    background-color: #FCE4DF !important;
    height: 0px !important;
}

.block-container {
    padding-top: 1rem !important;
}

[data-testid="stSidebar"] {
    background-color: #FCE4DF !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #3D2929 !important;
}

h1, h2, h3 {
    color: #A83232 !important;
}

p, label {
    color: #4A3030 !important;
}

.stTextInput input {
    background-color: #FFF4F1 !important;
    color: #3D2929 !important;
    border: 1px solid #D98B8B !important;
    border-radius: 8px !important;
}

.stNumberInput input {
    background-color: #FFF4F1 !important;
    color: #3D2929 !important;
    border: 1px solid #D98B8B !important;
    border-radius: 8px !important;
}

[data-baseweb="select"] > div {
    background-color: #FFF4F1 !important;
    color: #3D2929 !important;
    border-color: #D98B8B !important;
}

[data-testid="stFileUploader"] {
    background-color: #FFF4F1 !important;
    border-radius: 8px !important;
    border: 1px solid #D98B8B !important;
}

.stButton > button {
    background-color: #C94C4C !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

.stButton > button:hover {
    background-color: #A83232 !important;
    color: white !important;
}

input[type="radio"] {
    accent-color: #C94C4C !important;
}

hr {
    border-color: #D98B8B !important;
}

[data-testid="stRadio"] label {
    color: #3D2929 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# API KEY
# =========================================================

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

with open("recipes.json", "r") as f:
    recipes = json.load(f)

with open("substitutions.json", "r") as f:
    substitutions = json.load(f)

with open("unit_conversions.json", "r") as f:
    UNIT_CONVERSIONS = json.load(f)

with open("sample_pantry.json", "r") as f:
    pantry = json.load(f)


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=api_key
)


# =========================================================
# VECTOR STORE
# =========================================================

@st.cache_resource
def create_vectorstore():

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    documents = []

    for recipe in recipes:

        ingredients = recipe.get(
            "ingredients",
            []
        )

        ingredient_text = ", ".join(
            [
                (
                    f"{item.get('quantity', '')} "
                    f"{item.get('unit', '')} "
                    f"{item.get('item', '')}"
                )
                if isinstance(item, dict)
                else str(item)
                for item in ingredients
            ]
        )

        steps = recipe.get(
            "steps",
            recipe.get("instructions", [])
        )

        if isinstance(steps, list):
            steps_text = "\n".join(
                str(step)
                for step in steps
            )
        else:
            steps_text = str(steps)

        dietary_tags = recipe.get(
            "dietary_tags",
            recipe.get("diet", [])
        )

        if isinstance(dietary_tags, list):
            dietary_tags = ", ".join(
                str(tag)
                for tag in dietary_tags
            )

        text = f"""
Recipe: {recipe.get("name", "")}

Cuisine: {recipe.get("cuisine", "")}

Servings: {recipe.get("servings", "")}

Time: {recipe.get("time", "")}

Dietary tags: {dietary_tags}

Ingredients:
{ingredient_text}

Steps:
{steps_text}
"""

        documents.append(text)

    vectorstore = Chroma(
        collection_name="recipe_collection",
        embedding_function=embeddings,
        persist_directory="chroma_db"
    )

    if vectorstore._collection.count() == 0:
        vectorstore.add_texts(documents)

    return vectorstore


vectorstore = create_vectorstore()


# =========================================================
# TOOLS
# =========================================================

@tool
def ingredient_substitution(ingredient: str) -> str:
    """Find substitutes for an ingredient."""

    ingredient = ingredient.lower().strip()

    substitutes = substitutions.get(
        ingredient
    )

    if substitutes:
        return ", ".join(substitutes)

    return "No substitution found for this ingredient."


@tool
def unit_conversion(
    value: float,
    from_unit: str,
    to_unit: str
) -> str:
    """Convert cooking measurements."""

    key = (
        f"{from_unit.lower()}_to_"
        f"{to_unit.lower()}"
    )

    for category in [
        "volume",
        "weight"
    ]:

        if key in UNIT_CONVERSIONS.get(
            category,
            {}
        ):

            result = (
                value *
                UNIT_CONVERSIONS[
                    category
                ][key]
            )

            return str(result)

    return "No conversion found for these units."


# =========================================================
# MEMORY
# =========================================================

if "memory" not in st.session_state:

    st.session_state.memory = InMemorySaver()


if "agent" not in st.session_state:

    st.session_state.agent = create_agent(
        llm,
        tools=[
            ingredient_substitution,
            unit_conversion
        ],
        system_prompt=(
            "You are a simple recipe assistant. "
            "Remember the user's dietary preferences "
            "during the conversation."
        ),
        checkpointer=st.session_state.memory
    )


agent = st.session_state.agent

config = {
    "configurable": {
        "thread_id": "recipe-demo"
    }
}


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

class PantryItem(BaseModel):
    name: str
    quantity: str | None = None


class ReceiptExtract(BaseModel):
    store: str | None = None
    items: list[PantryItem]


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#A83232;'>AI Recipe & Pantry Helper</h2>",
        unsafe_allow_html=True
    )

    st.write("Select a feature")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Ask Recipe Question",
            "Find Recipes",
            "Ingredient Substitution",
            "Unit Conversion",
            "Receipt & Pantry",
            "Dietary Preferences"
        ],
        label_visibility="collapsed"
    )


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title("AI Recipe & Pantry Helper")

    st.write(
        "Your AI-powered cooking assistant."
    )

    st.write(
        "Use the menu on the left to select a feature."
    )

    st.divider()

    st.subheader("What you can do")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "**Ask Recipe Question**"
        )

        st.write(
            "Ask questions about recipes using recipe data."
        )

        st.markdown(
            "**Find Recipes**"
        )

        st.write(
            "Search recipes using semantic search."
        )

        st.markdown(
            "**Ingredient Substitution**"
        )

        st.write(
            "Find alternatives for ingredients."
        )

    with col2:

        st.markdown(
            "**Unit Conversion**"
        )

        st.write(
            "Convert common cooking measurements."
        )

        st.markdown(
            "**Receipt & Pantry**"
        )

        st.write(
            "Read grocery receipts and view pantry items."
        )

        st.markdown(
            "**Dietary Preferences**"
        )

        st.write(
            "Save your dietary preference for the conversation."
        )


# =========================================================
# ASK RECIPE QUESTION
# =========================================================

elif page == "Ask Recipe Question":

    st.title("Ask a Recipe Question")

    st.write(
        "Ask questions about the recipes in the collection."
    )

    question = st.text_input(
        "Your question",
        placeholder="Example: How do I make hummus?"
    )

    if st.button(
        "Ask Recipe Assistant",
        use_container_width=True
    ):

        if question:

            retrieved_docs = vectorstore.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            preference = st.session_state.get(
                "dietary_preference",
                "None"
            )

            prompt = f"""
You are an AI recipe assistant.

Answer the user's question using ONLY
the recipe information provided below.

If the answer is not available in the
recipe collection, politely say that
the information is not available.

User dietary preference:
{preference}

Recipe information:
{context}

User question:
{question}

Give a simple and helpful answer.
"""

            response = llm.invoke(
                prompt
            )

            st.subheader("Answer")

            st.write(
                response.content
            )

        else:

            st.warning(
                "Please enter a question."
            )


# =========================================================
# FIND RECIPES
# =========================================================

elif page == "Find Recipes":

    st.title("Find Recipes")

    st.write(
        "Describe what you want to cook."
    )

    recipe_query = st.text_input(
        "Recipe search",
        placeholder="Example: something spicy with chickpeas"
    )

    if st.button(
        "Search Recipes",
        use_container_width=True
    ):

        if recipe_query:

            results = vectorstore.similarity_search(
                recipe_query,
                k=5
            )

            preference = st.session_state.get(
                "dietary_preference",
                ""
            ).lower()

            shown = 0

            for doc in results:

                content = doc.page_content.lower()

                if preference == "vegan":
                    if "vegan" not in content:
                        continue

                if preference == "vegetarian":
                    if (
                        "vegetarian" not in content
                        and "vegan" not in content
                    ):
                        continue

                st.markdown(
                    doc.page_content
                )

                st.divider()

                shown += 1

            if shown == 0:

                st.write(
                    "No recipes matching your preference were found."
                )

        else:

            st.warning(
                "Please describe what you want to cook."
            )


# =========================================================
# INGREDIENT SUBSTITUTION
# =========================================================

elif page == "Ingredient Substitution":

    st.title("Ingredient Substitution")

    st.write(
        "Find alternatives for an ingredient."
    )

    ingredient = st.text_input(
        "Ingredient",
        placeholder="Example: butter"
    )

    if st.button(
        "Find Substitute",
        use_container_width=True
    ):

        if ingredient:

            result = ingredient_substitution.invoke(
                {
                    "ingredient": ingredient
                }
            )

            st.subheader("Substitutes")

            st.write(
                result
            )

        else:

            st.warning(
                "Please enter an ingredient."
            )


# =========================================================
# UNIT CONVERSION
# =========================================================

elif page == "Unit Conversion":

    st.title("Unit Conversion")

    st.write(
        "Convert cooking measurements."
    )

    value = st.number_input(
        "Value",
        min_value=0.0,
        value=1.0
    )

    col1, col2 = st.columns(2)

    with col1:

        from_unit = st.selectbox(
            "From",
            [
                "cup",
                "tbsp",
                "tsp",
                "oz",
                "lb",
                "kg"
            ]
        )

    with col2:

        to_unit = st.selectbox(
            "To",
            [
                "ml",
                "tbsp",
                "tsp",
                "g"
            ]
        )

    if st.button(
        "Convert",
        use_container_width=True
    ):

        result = unit_conversion.invoke(
            {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit
            }
        )

        st.subheader("Result")

        st.success(
            f"{value} {from_unit} = "
            f"{result} {to_unit}"
        )


# =========================================================
# RECEIPT & PANTRY
# =========================================================

elif page == "Receipt & Pantry":

    st.title("Receipt & Pantry")

    st.write(
        "Upload a grocery receipt to extract the items."
    )

    uploaded_file = st.file_uploader(
        "Upload receipt",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            width=500
        )

        if st.button(
            "Read Receipt",
            use_container_width=True
        ):

            buffer = io.BytesIO()

            image.save(
                buffer,
                format="PNG"
            )

            image_b64 = base64.b64encode(
                buffer.getvalue()
            ).decode()

            vision = ChatGroq(
                model="qwen/qwen3.8-27b",
                temperature=0,
                api_key=api_key,
                reasoning_format="parsed"
            )

            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/png;base64,"
                            f"{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """
Read this grocery receipt.

Return ONLY valid JSON with:

{
    "store": "store name",
    "items": [
        {
            "name": "item name",
            "quantity": "quantity"
        }
    ]
}
"""
                    }
                ]
            )

            response = vision.invoke(
                [message]
            )

            receipt_text = (
                response.content
                .replace(
                    "```json",
                    ""
                )
                .replace(
                    "```",
                    ""
                )
                .strip()
            )

            try:

                receipt_data = json.loads(
                    receipt_text
                )

                receipt_items = []

                for item in receipt_data.get(
                    "items",
                    []
                ):

                    receipt_items.append(
                        PantryItem(
                            name=str(
                                item.get(
                                    "name",
                                    ""
                                )
                            ),
                            quantity=str(
                                item.get(
                                    "quantity",
                                    ""
                                )
                            )
                        )
                    )

                extracted_receipt = ReceiptExtract(
                    store=receipt_data.get(
                        "store"
                    ),
                    items=receipt_items
                )

                st.subheader(
                    "Receipt Details"
                )

                st.write(
                    f"Store: "
                    f"{extracted_receipt.store}"
                )

                st.write("Items:")

                for item in extracted_receipt.items:

                    st.write(
                        f"{item.name} "
                        f"({item.quantity})"
                    )

            except Exception:

                st.error(
                    "Could not parse the receipt."
                )

                st.write(
                    receipt_text
                )

    st.divider()

    st.subheader("Current Pantry")

    for item in pantry.get(
        "have",
        []
    ):

        st.write(item)


# =========================================================
# DIETARY PREFERENCES
# =========================================================

elif page == "Dietary Preferences":

    st.title("Dietary Preferences")

    st.write(
        "Save your dietary preference for the current conversation."
    )

    dietary_preference = st.text_input(
        "Dietary preference",
        placeholder="Example: vegan, vegetarian, no nuts"
    )

    if st.button(
        "Save Preference",
        use_container_width=True
    ):

        if dietary_preference:

            st.session_state[
                "dietary_preference"
            ] = dietary_preference

            st.success(
                "Dietary preference saved."
            )

        else:

            st.warning(
                "Please enter a dietary preference."
            )

    if "dietary_preference" in st.session_state:

        st.divider()

        st.subheader(
            "Current Preference"
        )

        st.write(
            st.session_state[
                "dietary_preference"
            ]
        )

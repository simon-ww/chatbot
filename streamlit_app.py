import streamlit as st
import pandas as pd
import openai
import numpy as np
from flag import flag
import plotly.express as px
import pycountry
import json
from questions import questions
from sidebar import render_sidebar
from streamlit_extras.add_vertical_space import add_vertical_space

# Set up OpenAI API key
openai.api_key = st.secrets["secrets"]["OPENAI_API_KEY"]

st.set_page_config(page_title="Big Five", layout="wide")

def calculate_big_five_scores(responses):
    """Calculate Big Five scores based on user responses."""
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    scores = {trait: sum(responses[i::5]) / len(responses[i::5]) for i, trait in enumerate(traits)}
    return scores

def rescale_t_scores_to_one_five(t_scores):
    """Rescale T-scores to a 1-5 scale."""
    t_min = t_scores.min()
    t_max = t_scores.max()
    return 1 + 4 * (t_scores - t_min) / (t_max - t_min)

def calculate_similarity(user_scores, country_scores):
    """Calculate similarity between user scores and country scores using Euclidean distance."""
    user_vector = np.array([user_scores[trait] for trait in user_scores.keys()])
    country_vector = np.array([country_scores[trait] for trait in user_scores.keys()])
    distance = np.linalg.norm(user_vector - country_vector)  # Euclidean distance
    return distance

def alpha2_to_alpha3(alpha2):
    """Convert Alpha-2 country code to Alpha-3."""
    try:
        return pycountry.countries.get(alpha_2=alpha2).alpha_3
    except AttributeError:
        return None

def get_flag_emoji(country_code):
    """Get the country flag emoji using the `flag` package."""
    try:
        return flag(country_code.upper())
    except Exception:
        return "❓"

def get_negotiation_advice(scores):
    """Use OpenAI to provide negotiation advice based on personality scores."""
    personality_profile = ", ".join([f"{trait}: {score:.2f}" for trait, score in scores.items()])
    prompt = f"Based on this personality profile: {personality_profile}, provide negotiation advice tailored to this person's strengths and weaknesses."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a negotiations coach that provides advice on how personality traits and cultural background impacts negotiation tactics and strategies"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while fetching negotiation advice: {e}")
        return "Unable to generate advice due to an error."

def render_country_table(df):
    """Render a country table with emoji flags."""
    table_html = "<table>"
    table_html += "<tr><th>Country</th><th>Similarity</th><th>Flag</th></tr>"
    for _, row in df.iterrows():
        table_html += (
            f"<tr>"
            f"<td>{row['country']}</td>"
            f"<td>{row['similarity']:.2f}</td>"
            f"<td>{get_flag_emoji(row['country_code'])}</td>"
            f"</tr>"
        )
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

def render_similarity_map(similarity_df):
    """Render a choropleth map based on similarity scores."""

    # Load GeoJSON file 
    with open("countries.geo.json", "r") as geojson_file:
        geojson_data = json.load(geojson_file)

    # Convert Alpha-2 to Alpha-3 codes
    similarity_df['country_code_alpha_3'] = similarity_df['country_code'].apply(alpha2_to_alpha3)

    map_data = similarity_df.rename(columns={"country_code_alpha_3": "iso_alpha", "similarity": "Similarity"})

    fig = px.choropleth(
        map_data,
        geojson=geojson_data,
        locations="iso_alpha",
        color="Similarity",
        hover_name="country",
        hover_data=["Similarity"],
        color_continuous_scale="Viridis",
    )
    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#e5e5e5",  # Coastline color for contrast
        showland=True,
        landcolor="#252323",  # Land color
        showocean=True,
        oceancolor="#0e1117",  # Ocean color
        fitbounds="locations"
    )
    fig.update_layout(
        autosize=False,
        paper_bgcolor="#0e1117",  # Background color
        plot_bgcolor="#0e1117",  # Plot area background color
        font_color="white",  # Font color for text
        margin=dict(l=10, r=10, t=1, b=1),
        coloraxis_colorbar={
            "title": "Similarity",
            "tickfont": {"color": "white"},  # Tick font color
            "titlefont": {"color": "white"}  # Title font color
        }
    )
    st.plotly_chart(fig, use_container_width=True) ## CREATE MAP GRAPH

# Inside compare_with_dataset function
def compare_with_dataset(scores, dataset_path):
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Normalize column names in the dataset
        df.columns = [col.lower() for col in df.columns]

        # Check if the required columns exist
        t_score_columns = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        if not all(col in df.columns for col in t_score_columns + ["country", "country_code"]):
            raise ValueError("Dataset is missing required columns, including 'country' and 'country_code'.")

        # Drop rows with missing values in Big Five columns
        df = df.dropna(subset=t_score_columns)

        # Ensure the `country_code` column contains valid country codes
        if "country_code" in df.columns:
            df["country_code"] = df["country_code"].apply(lambda x: str(x).strip().upper() if isinstance(x, str) else "UNKNOWN")
        else:
            raise ValueError("Dataset is missing the 'country_code' column.")

        # Extract relevant Big Five columns and rescale them
        df_rescaled = df[t_score_columns].apply(rescale_t_scores_to_one_five)
        df_rescaled = pd.concat([df_rescaled, df[["country", "country_code"]].reset_index(drop=True)], axis=1)  # Add country and country_code columns

        # Convert user scores to lowercase keys for comparison
        scores = {k.lower(): v for k, v in scores.items()}

        # Calculate similarity scores for each country
        similarities = []
        for _, row in df_rescaled.iterrows():
            country_scores = row[t_score_columns].to_dict()
            similarity = calculate_similarity(scores, country_scores)
            similarities.append({"country": row["country"], "country_code": row["country_code"], "similarity": similarity})

        # Convert to DataFrame and sort by similarity
        similarity_df = pd.DataFrame(similarities)
        similarity_df = similarity_df.sort_values(by="similarity", ascending=True)  # Lower distance = more similar

        # DROPDOWN FOR PERSONALITY TRAIT
        #st.markdown("<h3 style='text-align: center;'>Explore Countries by Personality Traits</h3>", unsafe_allow_html=True)
        trait = st.selectbox("Select a personality trait to explore the top 5 similar countries:", ["Combined","Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"])

        # Create a centered container for tables
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 1, 0.2, 4, 0.1)
)
        # Generate cultural analysis of the most similar country
        with row0_1:
            st.markdown(
                f"<h3 style='text-align: center;'>You're most like {similarity_df.loc[similarity_df['similarity'].idxmin(), 'country']} {flag(similarity_df.loc[similarity_df['similarity'].idxmin(), 'country_code'])}</h3>",
                unsafe_allow_html=True
            )
            st.markdown("<p style='text-align: center;'>5 Most Similar Countries:</p>", unsafe_allow_html=True)
            
        with row0_1: 
            render_country_table(similarity_df.head(5))
            cultural_analysis = generate_country_analysis(similarity_df)
        with row0_2:
             add_vertical_space()

        with row0_2:
            st.markdown(
                f"<div style='text-align: center;'>{cultural_analysis}</div>",
                unsafe_allow_html=True
            )

        row0_spacer3, row0_3, row0_spacer4, row0_4, row0_spacer5 = st.columns(
        (0.1, 1, 0.2, 4, 0.1)
)
        with row0_3:
            st.markdown(
                f"<h2 style='text-align: center;'>You're least like {similarity_df.loc[similarity_df['similarity'].idxmax(), 'country']} {flag(similarity_df.loc[similarity_df['similarity'].idxmax(), 'country_code'])}</h2>",
                unsafe_allow_html=True
            )
            st.markdown("<p style='text-align: center;'>5 Least Similar Countries:</p>", unsafe_allow_html=True)
            render_country_table(similarity_df.tail(5))
            cultural_disimilar = generate_disimilar(similarity_df)
        with row0_4:
             add_vertical_space()
        with row0_4:
            st.markdown(
                f"<div style='text-align: center;'>{cultural_disimilar}</div>",
                unsafe_allow_html=True
            )

        # Create the map visualization
        spacer4, col3, spacer5 = st.columns([1, 6, 1])
        with col3:
            st.markdown("<h3 style='text-align: center;'>Explore the Rest of the World</h3>", unsafe_allow_html=True)
            render_similarity_map(similarity_df)

    except FileNotFoundError:
        st.error(f"Dataset not found at {dataset_path}. Please upload the correct file.")
    except ValueError as ve:
        st.error(f"Dataset format error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# MOST SIMILAR COUNTRY
def generate_country_analysis(similarity_df):
    """Analyze negotiation cultural norms for the most similar country."""
    # Find the most similar country
    most_similar_country = similarity_df.loc[similarity_df['similarity'].idxmin()]
    most_country = most_similar_country['country']
    prompt = f"""
    Provide a short blurb less than 150 words that very briefly summarizes negotiation cultural norms specific to {most_country}, as well as one very interesting fact about negotiation or communication norms/styles there.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a negotiations coach that provides advice on how personality traits and cultural background impacts negotiation tactics and strategies."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating cultural analysis: {e}")
        return "Unable to generate cultural analysis at this time."

# LEAST SIMILAR COUNTRY
def generate_disimilar(similarity_df):
    """Analyze negotiation cultural norms for the least similar country."""
    # Find the LEAST similar country
    least_similar_country = similarity_df.loc[similarity_df['similarity'].idxmax()]
    least_country = least_similar_country['country']
    prompt = f"""
    Provide a short blurb less than 150 words that very briefly summarizes negotiation cultural norms specific to {least_country}, as well as one very interesting fact about negotiation or communication norms/styles there
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a negotiations coach that provides advice on how personality traits and cultural background impacts negotiation tactics and strategies."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating cultural analysis: {e}")
        return "Unable to generate cultural analysis at this time."

#########################################################
#################### Streamlit App ######################
#########################################################
def main():
    st.markdown("<h1 style='text-align: center;'>🥧 The Pie's not always Apple</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Want to take your newly minted negotiations skills global?</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Try this tool to understand how your personality helps (or hurts) in cross-culture negotiations.</p>", unsafe_allow_html=True)
    st.markdown(
             """ 
             <div style='text-align: center;'>
             To excel as a negotiator, especially in an increasingly globalized world, one must embrace the diversity of cultural norms and personality traits. As Erin Meyer highlights in <em>Getting to Sí, Ja, Oui, Hai, and Da</em> what drives a deal forward in one culture can derail it in another​. For instance, while open disagreement is seen as a constructive dialogue in cultures like Germany or Israel, it may shut down discussions in Mexico or Japan. Similarly, building trust may involve professional competency in the U.S. but require deep personal relationships in China​. 
             
             Understanding and adapting to these nuances enables negotiators to decode subtle signals and avoid miscommunication. Whether it’s gauging emotional expressiveness, tailoring communication to the right level of formality, or recognizing when “yes” means “maybe,” being culturally attuned is essential to fostering trust and mutual understanding​. By broadening your perspective and honing your cultural intelligence, you pave the way for more successful outcomes and lasting partnerships.
             
             This tool leverages data analytics and a CustomGPT to dynamically provide you personality results, contextualized against how they align with that of generalized results from other countries. 
             </div>
             """, unsafe_allow_html=True)

    # Render the sidebar and get the user's navigation choice
    with st.sidebar:
        st.title("🥧 Culture Pie")
        st.subheader("Negotiations - Fall 2024")
        st.markdown(
            """Culture pie is a tool built for exploring the nusances created by the intersection of personality and culture when people of different backgrounds come to the table to negotiate."""
        )
        st.markdown("<p style='text-align: center;'>Simon Chen WG'26 <a href='simoncn@wharton.upenn.edu' target='_blank'>simoncn@wharton.upenn.edu</a>.</p>",unsafe_allow_html=True)
        st.subheader("Works Cited")
        st.markdown("""
        - [IPIP, Big Five Personality Test](https://ipip.ori.org/)
        - [Meyer, Erin. “Getting to Sí, Ja, Oui, Hai, and Da.” Harvard Business Review, Dec. 2015](https://hbr.org/2015/12/getting-to-si-ja-oui-hai-and-da)
        - Fisher, Roger, and William Ury. Getting to Yes: Negotiating Agreement Without Giving In
        """)
        st.subheader("Data Set")
        st.markdown("""
        - [The EcoCultural Dataset, OSF](https://osf.io/r9msf/)
        """)
        st.markdown(
            """["The Geographic Distribution of Big Five Personality Traits"](https://www.researchgate.net/publication/260244540_The_Geographic_Distribution_of_Big_Five_Personality_Traits_Patterns_and_Profiles_of_Human_Self-Description_Across_56_Nations) is part of the International Sexuality Description Project (ISDP) and collected data on the Big Five personality traits using the 44-item Big Five Inventory (BFI). The data set spans responses from 17,837 participants across 56 nations, representing 10 geographic world regions, 29 languages, and six continents. Most samples were composed of college students, with some community-based participants, and were convenience samples."""
        )
        st.markdown("""
        Key features of the data collection:
        - Translations: The BFI was translated into 28 languages using translation and back-translation methods to ensure cultural and linguistic appropriateness.
        - Sampling: Participants were mainly volunteers, with some receiving incentives or course credit.
        - Methodology: The study utilized self-reported measures with a high return rate among college students (~95%) but lower among community samples (~50%).

        """)

    # Initialize session state variables
    if "test_submitted" not in st.session_state:
        st.session_state.test_submitted = False
    if "advice_generated" not in st.session_state:
        st.session_state.advice_generated = False

    # Step 1: Big Five Questionnaire
    with st.expander("First - a few questions to get to know you", expanded=True):
        responses = {}
        
        # Display questions with sliders grouped by trait
        for trait, trait_questions in questions.items():
            #st.subheader(trait)
            st.caption("Here are a number of characteristics that may or may not apply to you. For example, do you agree that you are someone who likes to spend time with others? Please write a number next to each statement to indicate the extent to which you agree or disagree with that statement. Rate on a scale of 1 (Strongly Disagree) to 5 (Strongly Agree).")
            responses[trait] = []
            for question in trait_questions:
                response = st.slider(question, 1, 5, 3, key=f"{trait}_{question}")
                responses[trait].append(response)

        # Step 1 Button: Submit Test
        if st.button("Submit Test"):
            st.session_state.test_submitted = True
            st.session_state.advice_generated = False  # Reset advice state when test is resubmitted

    if st.session_state.test_submitted:
        # Flatten responses for scoring
        flattened_responses = [response for trait_responses in responses.values() for response in trait_responses]

        # Step 2: Calculate Scores
        scores = calculate_big_five_scores(flattened_responses)
        st.markdown("<h2 style='text-align: center;'>Your Big Five Personality Results", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>The “Big Five” are five broad dimensions meant to capture the range of human personality. Think of them like those quizzes on Buzzfeed that promise to tell you “what type of person you are,” but this is actually real. It’s the best, most rigorous way scientists have come up with to capture the variance in human personality.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>One should be very wary of using canned “norms” because it isn’t obvious that one could ever find a population of which one’s present sample is a representative subset. Most “norms” are misleading, and therefore they should not be used.</p>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>More on Norms from IPIP <a href='https://ipip.ori.org/newNorms.htm' target='_blank'>our resource page</a>.</p>",
            unsafe_allow_html=True
        )

        # Create a bar graph with distinct colors for each bar
        traits = list(scores.keys())
        values = list(scores.values())
        colors = ["#C2185B", "#7C4DFF", "#536DFE", "#00BCD4", "#64FFDA"]  # Example color palette
        fig = px.bar(
            x=traits,
            y=values,
            labels={"x": "Traits", "y": "Score"},
            title="Your Big Five Personality Profile",
            color=traits,  # Add a color dimension
            color_discrete_sequence=colors  # Assign custom colors
        )
        fig.update_layout(
            xaxis_title="Traits",
            yaxis_title="Score",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Read more about Big Five Traits", expanded=False):
            st.markdown("##### Big Five Traits")
            st.markdown("<h4 style='text-align: center;'>🙂‍↕️ Agreeableness</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Agreeableness is a measure of courteousness, flexibility, sympathy, trust, cooperation, and tolerance. Agreeable people are kind, warm, altruistic, and tend to be both trusting and trustworthy. They value relationships and avoid conflict. Research has found that agreeable individuals have greater motivation to achieve interpersonal intimacy, which should lead to less assertive tactics in a negotiation setting. Their tendency to be trusting and cooperative might prove constructive. It could even promote the positive negotiation processes needed to achieve economic joint gain. But that success may come at the expense of individual economic outcomes in the face of a competitive counterpart. Higher levels of agreeableness have been found to be associated with a greater susceptibility to anchoring. And for sellers (but not buyers), agreeableness is associated with lower gains (even controlling for anchoring effects).</p>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>🥳 Extraversion</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Extraversion represents the tendency to be sociable, dominant, assertive, gregarious, confident, and positive (Costa & McCrae, 1992; Watson & Clark, 1997). Extraverts tend to have more friends and spend more time in social situations than do introverts. Because of their sociable nature such individuals may disclose more information about their own preferences and alternatives to agreement during a negotiation. That tendency could be disadvantageous in a highly competitive context. But these same sociable traits may be an asset for integrative bargaining that requires more communication and social interaction to reveal hidden trade-offs and compatibilities (Barry & Friedman,1998). Even so, the assertiveness sub-component could help negotiators stand their ground (Elfenbein, Curhan, Eisenkraft, Shirako, & Brown, 2010). By contrast, the anxiety that introverts feel during social encounters may lead them to make concessions that enable exit from the situation. Finally, extraversion could facilitate the rapport building needed to establish subjective value for both the self and counterpart, although note that extraversion increases one’s susceptibility to anchoring effects.</p>", unsafe_allow_html=True)

            st.markdown("<h4 style='text-align: center;'>🥹 Conscientiousness</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Conscientiousness is a measure of self-discipline, indicating that individuals are well organized, careful, responsible, and motivated to achieve (Costa & McCrae, 1992; John & Srivastava, 1999). Of the five fundamental personality traits captured by the TIPI, conscientiousness is the best predictor of overall job performance across a wide array of occupations (Barrick & Mount, 1991). Conscientious negotiators may outperform their less conscientious peers, given their generally greater task achievement and thorough preparation for complex tasks. Furthermore, highly conscientious individuals may facilitate an overall negotiation experience that stays focused on the task instead of personal rancor.</p>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>🤗 Openness</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Openness is a measure of imaginativeness, broad-mindedness, and divergent thinking, describing people who are intellectually curious, creative, resourceful, and willing to consider unconventional ideas (Costa & McCrae,1992; John & Srivastava, 1999). Highly open negotiators might approach the unstructured task with greater flexibility and willingness to pursue creative strategies towards more integrative deals (Barry & Friedman, 1998). Open negotiators might be less prone to the ‘‘fixed pie bias,’’ whereby individuals assume that their own and their counterpart’s preferences are diametrically opposed. Their greater flexibility and divergent thinking could help open negotiators to craft better deals for themselves and others. </p>", unsafe_allow_html=True)            
            st.markdown("<h4 style='text-align: center;'>🫨 Neuroticism</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Neuroticism, the inverse of emotional stability, refers to a general level of anxiety, depression, worry, and insecurity (Costa & McCrae,1992; John & Srivastava, 1999). It involves a greater tendency to experience negative affect such as fear, sadness, guilt, and anger. Neurotics are more anxious, moody, prone to emotional distress, and more sensitive to negative stimuli, such as the stimuli involved with the uncertain process of negotiating. Neurotic negotiators may struggle to engage fully with the task and their relationship partners, likely resulting in less optimal economic and psychological outcomes.</p>", unsafe_allow_html=True) 
    

    if st.session_state.test_submitted:
        # Flatten responses for scoring
        flattened_responses = [response for trait_responses in responses.values() for response in trait_responses]
        # Step 2: Calculate Scores
        scores = calculate_big_five_scores(flattened_responses)

        advice = get_negotiation_advice(scores)
        st.write(advice)
        st.write("***Here is your personalized negotiation recommendations based on your Big Five personality test results. The results are generated by a customGPT AI bot built upon the OpenAI LLM platform****")

        # Step 3: Compare with Dataset
        st.markdown("<h2 style='text-align: center;'>How your Traits Translate", unsafe_allow_html=True)
        dataset_path = "/workspaces/chatbot/data/global_bigfive_data.csv"  # Replace with your actual dataset path
        compare_with_dataset(scores, dataset_path)

    with st.expander("Find Tactics & Strategies to match your Personality & Context", expanded=False):
     
            st.markdown("<h4 style='text-align: center;'>Responsiveness</h4>", unsafe_allow_html=True)
            st.markdown("""is a powerful tactic in negotiations that involves demonstrating attentiveness and support to the other party's needs, concerns, and emotions. This approach can significantly enhance the negotiation process by building trust, fostering positive relationships, and creating a more collaborative atmosphere. Responsiveness consists of three key elements:""")
            st.markdown("""
                            - **Care** component of responsiveness involves demonstrating genuine concern for the other party's well-being and interests. This can be expressed through: empathy and emotional support, willingness to accommodate the other party's needs when possible, commitment to finding mutually beneficial solutions. 
                            - **Validation** involves acknowledging and accepting the other party's experiences, emotions, and viewpoints as legitimate and worthy of consideration. This element of responsiveness includes: recognizing the validity of the other party's perspective, showing appreciation for their abilities and traits, and respecting their worldview, even if it differs from your own. 
                            - **Understanding** in responsiveness refers to accurately perceiving and comprehending the other party's perspective, including their: core values and beliefs, immediate thoughts and feelings, and underlying needs and interests. 
                            """)
            
            st.markdown("<h4 style='text-align: center;'>Distancing Yourself</h4>", unsafe_allow_html=True)
            st.markdown("""***Distance Yourself*** tactics in negotiations are strategies used to gain perspective and maintain emotional control during challenging discussions. These techniques help negotiators step back from the immediate situation, allowing for more objective analysis and decision-making. These distancing tactics share a common goal: to help negotiators maintain composure, think more clearly, and make better decisions during challenging conversations. By creating psychological distance, negotiators can avoid reactive responses, consider multiple perspectives, and focus on long-term objectives rather than getting caught up in the heat of the moment. Here are three specific tactics:""")
            st.markdown("""
                            - **Fly on the Wall**: The "fly on the wall" technique involves observing a situation as an unobtrusive, neutral party without interfering or influencing the events. This technique allows negotiators to gain a more impartial view of the situation, potentially revealing insights that might be missed when deeply engaged in the discussion. 
                            - **Peering Down from a Balcony**: The "balcony strategy," popularized by William Ury, involves mentally stepping back from the negotiation, as if observing it from a balcony. 
                            - **Refer to Self in Third Person**: is another distancing tactic that can be effective in negotiations. 
                            """)
            
            st.markdown("<h4 style='text-align: center;'>Strategic Display</h4>", unsafe_allow_html=True)
            st.markdown("""is a tactic used in negotiations to intentionally convey specific information, emotions, or behaviors to influence the other party's perceptions and decisions. This approach involves carefully crafting one's presentation and communication to achieve desired outcomes in the negotiation process. Strategic display, when used effectively, can be a powerful tool in shaping the negotiation environment and influencing outcomes. However, it's important to use this tactic ethically and in conjunction with other negotiation strategies for optimal results. The Key Elements are:
                        """)
            st.markdown("""
                            - **Controlled Information Sharing**: Selectively revealing or withholding information to shape the other party's understanding of the situation.
                            - **Emotional Management**: Deliberately expressing or suppressing emotions to elicit specific responses from counterparts.
                            - **Behavioral Positioning**: Adopting particular behaviors or stances to project strength, flexibility, or other desired qualities.
                            """)
            st.markdown("<h4 style='text-align: center;'>Self-disclosure:</h4>", unsafe_allow_html=True)
            st.markdown("""is a strategic tactic in negotiations that involves intentionally sharing personal information, thoughts, or feelings with the other party. When used effectively, it can significantly impact the negotiation process and outcomes. Benefits of Self-Disclosure in Negotiations:""")
            st.markdown("""
                 - Building Trust and Rapport
                 - Emotional Regulation
                 - Information Exchange
                 """)
            
            st.markdown("<h4 style='text-align: center;'>And of course,</h4>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center;'>grow the 🥧 and 🔪 the pie! Thanks Professor Cooney!</h4>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()

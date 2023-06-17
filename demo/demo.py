import streamlit as st
import pandas as pd
import numpy as np
import demo_utils
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import pandas as pd
import streamlit as st

# Intialize a list of tuples containing the CSS styles for table headers
th_props = [ ('text-align', 'left'),
            ('font-weight', 'bold'),('color', '#6d6d6d'),
            ('background-color', '#eeeeef'), ('border','1px solid #eeeeef'),
    ]

# Intialize a list of tuples containing the CSS styles for table data
td_props = [ ('text-align', 'left')]

# Define hover props for table data and headers
cell_hover_props = [('background-color', '#eeeeef')]
headers_props = [('text-align','center'), ('font-size','1.1em')]

# Aggregate styles in a list
styles = [
    dict(selector="th", props=th_props),
    dict(selector="td", props=td_props),
    dict(selector="td:hover",props=cell_hover_props),
    dict(selector='th.col_heading',props=headers_props),
    dict(selector='th.col_heading.level0',props=headers_props),
    dict(selector='th.col_heading.level1',props=td_props)
]

c1, c2= st.columns(2)

with c1:
    model_name = st.radio(
        "choose your model",
        ('model6.1-v2.2', 'model6-v2.2', 'model4-v2.2','model4-v2.1'))
    
with c2:
    df = pd.DataFrame([
        {'keyword': 'permanent', 'translation': '永久物'}
    ])
    df = st.data_editor(df, num_rows="dynamic")
    dic={x[0]:x[1] for x in df.to_numpy()}
    
model_name=model_name.replace('-','-T-')
T = demo_utils.load_translator(f'result/{model_name}.pt')
CT = demo_utils.create_card_translator(T, dic)

test_data = demo_utils.load_test_data('one')
key = st.selectbox('Choose a card in test set.', test_data.keys())
card = test_data[key]

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.markdown(
    """<style>
        .dataframe {text-align: left !important}
    </style>
    """, unsafe_allow_html=True)
df=pd.DataFrame(
    columns=("English rule text","Standard translation","Translator output"),
    data=np.array([card['src'].replace('\n', '<br>'),
                          card['trg'].replace(' ', '<br>'),
                          demo_utils.reformat(CT.translate(card['src'])),]).reshape(1,3),                        
)
st.markdown(df.style.set_table_styles(styles).to_html(escape=False),unsafe_allow_html=True)



# c1, c2, c3 = st.columns(3)
# with c1:
#     st.subheader('English rule text')
#     st.write(card['src'].replace('\n', '\n\n'))

# with c2:
#     st.subheader('Standard translation')
#     st.write(card['trg'].replace(' ', '\n\n'))

# with c3:
#     st.subheader('Translator output')
#     st.write(demo_utils.reformat(CT.translate(card['src'])))

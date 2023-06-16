import streamlit as st
import pandas as pd
import numpy as np
import demo_utils


df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))

st.dataframe(df)

# df = pd.DataFrame([
#     {'keyword': 'permanent', 'translation': '永久物'}
# ])
# df = st.data_editor(df, num_rows="dynamic")

# dic = {x[0]: x[1] for x in df}

dic = {}
T = demo_utils.load_translator('result/model6.1-T-v2.2.pt')
CT = demo_utils.create_card_translator(T, dic)

test_data = demo_utils.load_test_data('one')
key = st.selectbox('Choose a card in test set.', test_data.keys())
card = test_data[key]


c1, c2, c3 = st.columns(3)

with c1:
    st.subheader('English rule text')
    st.write(card['src'].replace('\n', '\n\n'))

with c2:
    st.subheader('Standard translation')
    st.write(card['trg'].replace(' ', '\n\n'))

with c3:
    st.subheader('Translator output')
    st.write(CT.translate(card['src']))

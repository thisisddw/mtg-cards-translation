import streamlit as st
import pandas as pd
import numpy as np
import demo_utils
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import pandas as pd
import streamlit as st
from PIL import Image
import os
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
st.markdown("""
<style>
.huge-font {
    font-size:32px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="huge-font">万智牌卡牌翻译</p>', unsafe_allow_html=True)
st.markdown("""
<style>
.big-font {
    font-size:24px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("本项目为南开大学20级本科生杜岱玮（2011421）和徐百朋（2012109）的语音信息处理课程期末作业，"
            +"这是一个交互性演示网页，我们的项目代码已经开源在github上，欢迎访问。https://github.com/thisisddw/mtg-cards-translation\n\n"
            +"如要了解更多信息，请查看我们的[实验报告](https://github.com/thisisddw/mtg-cards-translation/blob/master/MTGCARD_TRANSLATION.pdf)。",
            unsafe_allow_html=True)
st.markdown('<p class="big-font">简介</p>', unsafe_allow_html=True)

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
c1, c2= st.columns([1,2])
with c1:
    image=Image.open(ROOT_DIR+'\demo\mtgcard.jpg')
    st.image(image,caption='一张稀有的万智牌，中文名\"飘萍皇\"',use_column_width=True)
with c2:
    st.markdown('MTG是一款风靡全球的集换式卡牌游戏。本研究旨在构建一种机器翻译模型，它能够相对准确地将卡牌的英文描述翻译为中文描述。\n\n'
                +'我们在研究中使用不同架构和数据集训练了多个模型，实验报告中已有详细描述。下面提供了训练好的模型，您可以尝试对测试集中的卡牌进行翻译，观察不同模型的效果。')

'我们提供下面这些模型：'
df=pd.DataFrame(columns=("Model","Description","BLEU","Model Parameters"),
                data=np.array( [['model4-v2.1', 'RNNsearch on dataset v2.1', '64.96','11M'],
                                ['model4-v2.2', 'RNNsearch on dataset v2.2', '65.01','11M'],
                                ['model6-v2.2', 'Transformer(learnable position encoding) on dataset v2.2', '65.29','5.3M'],
                                ['model6.1-v2.2', 'Transformer(fixed position encoding) on dataset v2.2', '69.41','5.2M'],
                                ]))
st.dataframe(df,hide_index=True)

'''模型名称的格式为模型编号+训练数据集编号，例如model4-v2.1表示model4在数据集v2.1上训练的结果。
model4是采用RNNSearch架构的模型，model6是采用Transformer架构的模型。关于我们使用的模型和数据集的更多信息，
还请参阅[实验报告](https://github.com/thisisddw/mtg-cards-translation/blob/master/MTGCARD_TRANSLATION.pdf)。'''

# st.markdown('<p class="big-font">Build Dataset</p>', unsafe_allow_html=True)
# st.markdown("""
# 我们自己构建了数据集。包括数据预处理，划分数据集，数据增强三个部分
# - 数据处理：原始数据中，一条数据除了卡牌名和规则文字外，还包括了使用语言，所在系列，卡牌编号，发行版本编号等等，我们过滤无用信息，将其预处理为中英对照表的形式，以json格式存储。
# - 划分数据集：我们将数据集划分为训练集，验证集和测试集，测试集约30000条数据，验证集和测试集各有约500条数据。
# - 数据增强：我们使用"添加标签"方法，将训练集中的中文关键词替换为对应的英文关键词，并将所有关键词以及卡牌名用尖括号标出。这些修改后的数据将加入训练集中，以训练翻译器原封不动翻译卡牌名称和关键词的能力。
# 输出之后，在后处理阶段将尖括号去除，将尖括号内的英文翻译成中文，得到最终的翻译结果。

# """)

# st.markdown('<p class="big-font">模型简介</p>', unsafe_allow_html=True)
# st.markdown("""我们基于两种模型进行了实验，分别是RNNsearch和Transformer。下面是这两种模型的架构图。""")
c1, c2= st.columns(2)
with c1:
    image=Image.open(ROOT_DIR+'\demo\RNNsearch.png')
    st.image(image,caption='RNNsearch(model 4)',use_column_width=True)
with c2:
    image=Image.open(ROOT_DIR+'\demo\Transformer.png')
    st.image(image,caption='Transformer(model 6)',use_column_width=True)

st.markdown('<p class="big-font">模型特点</p>', unsafe_allow_html=True)
st.markdown("""您可能会在试验模型的过程中注意这些有趣特点：
- 基于Transformer的model6经常做出过于自由的翻译，引起语序混乱、语义丢失等问题。model6.1改用固定位置编码，这些问题有所缓解。
- 基于RNNSearch的model4经常产生包含重复词句的输出，我们认为这可能是RNN解码器导致的问题。model4-v2.1对字典替换表现不佳，这是v2.1数据集的缺陷导致的，我们的[实验报告](https://github.com/thisisddw/mtg-cards-translation/blob/master/MTGCARD_TRANSLATION.pdf)中有详细描述。
- model4和model6都会出现翻译错误的情况，但是model6的错误更加隐蔽，因为它的输出看起来更像是一句完整的话。

就说这么多，祝您玩得愉快！
""")

# st.markdown('<p class="big-font">模型对比</p>', unsafe_allow_html=True)

c1, c2= st.columns([1,1.2])

with c1:
    model_name = st.radio(
        "选择模型",
        ('model6.1-v2.2', 'model6-v2.2', 'model4-v2.2','model4-v2.1'))
    st.markdown(
    """<style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
    
with c2:
    st.markdown('<p class="big-font">添加字典项</p>', unsafe_allow_html=True)
    df = pd.DataFrame([
        {'keyword': 'permanent', 'translation': '永久物'},
        {'keyword': 'compleat', 'translation': '完化'},
    ])
    df = st.data_editor(df, num_rows="dynamic")
   
    dic={x[0]:x[1] for x in df.to_numpy()}
for k in list(dic.keys()):
    if k==None or dic[k]==None:
        del dic[k]
model_name=model_name.replace('-','-T-')
T = demo_utils.load_translator(f'result/{model_name}.pt')
CT = demo_utils.create_card_translator(T, dic)

test_data = demo_utils.load_test_data('one')
#change font of the selectbox

key = st.selectbox('选择要翻译的卡牌', test_data.keys(),)
st.markdown(
    """<style>
    div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
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
    columns=("英文卡牌规则","中文标准翻译","模型翻译结果"),
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

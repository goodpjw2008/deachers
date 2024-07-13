#from dotenv import load_dotenv
#load_dotenv()

# 랭체인관련 모듈들을 import
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.callbacks.base import BaseCallbackHandler

# streamlit 과 기타 모듈을 import
import streamlit as st
from PIL import Image
import datetime

# 실시간으로 답변이 나오는 효과를 위해 콜백함수 사용
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="성적을 부탁해 디처스",
        page_icon=":full_moon_with_face:",
        layout="wide"
    )

    # 제목 및 설명 표시
    st.write("### :full_moon_with_face: :blue[성적을 부탁해 디처스]")

    # 대화 내용 기억을 위해 히스토리 설정
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

    # 사이드바에 이미지 표시
    image_path = 'image/logo.jpg'  # 이미지 파일 경로
    image = Image.open(image_path)  # 이미지 로드
    st.sidebar.image(image, use_column_width=True)  # 사이드바에 이미지 표시

    with st.sidebar:
        # 사이드바에 입력 필드들 생성
        st.markdown("### :basketball: 시험이름")
        test_name = st.text_input('시험이름', '1학년 1학기 기말', label_visibility="collapsed")
        with st.expander("#### :soccer:  시험정보", expanded=False, ):
            st.markdown("##### :white_check_mark: 시험목표")
            test_objective = st.text_input('시험목표', '담임이 지켜보고 있다.', label_visibility="collapsed")
            st.markdown("##### :white_check_mark: 시험일자")
            test_day = st.date_input('시험 시작일자', datetime.date(2024, 8, 30),label_visibility="collapsed")
            st.markdown("##### :white_check_mark: 시험준비기간")
            test_prepare_day = st.slider("시험준비기간", 0, 50, 21, label_visibility="collapsed")
            st.markdown("##### :white_check_mark: 시험정보")
            file_path = 'data/test_info.txt'
            # 파일을 읽어서 변수에 저장
            with open(file_path, 'r', encoding='euc-kr') as file:
                test_info = file.read()
            test_information = st.text_area(
                "",
                test_info,
                height=1000,
                label_visibility="collapsed"
            )

    # 메시지 히스토리 표시
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
   

    # 시험 계획 버튼 클릭 시 실행
    if st.sidebar.button(":pencil: 디처스!! 시험계획을 세워줘", type="primary"):
        query = (
            f"시험계획을 세워줘.\n"
            f"이번에 보는 시험은 {test_name} 시험이야.\n"
            f"나의 시험목표는 {test_objective}로 정했어.\n"
            f"시험일자는 {test_day} 이야.\n"
            f"시험준비기간은 {test_prepare_day} 일 동안 공부할꺼야.\n"
            f"시험 정보는 다음과 같아.\n\n"
            f"{test_information}\n\n"
            f"내가 시험을 잘 볼 수 있게 시험계획을 세워줘. 목차를 아래와 같이 만들어줘."
            f"1. 시험계획명 : 계획이름은 시험목표인 {test_objective} 달성을 위한 이름으로 감동적이고 멋잇게 만들어줘. 크고 굵은 색으로 적어줘.\n"
            f"2. 시험전략 : 시험목표와 시험 정보를 분석해서 만들어줘.\n"
            f"3. 시험준비일정: 달력 형태로 만들어서 보여줘. 달력의 형태는 월화수목금토일 형태의 달력모양으로 표형태로 만들어줘. 표안에는 다음의 예시 처럼 넣어줘. (야자1) 수학 : 세부공부내용 n(야자2) 영어 : 세부공부내용"
        )
        st.chat_message("human").write("성적을 부탁해 디처스~~ 시험 계획을 세워줘!!")
        with st.spinner('답변중...'):
            pg = st.empty()
            response = do_ask(query, pg, msgs)  # ChatGPT에게 물어보는 부분
            st.chat_message("ai").write(response['output'])
            pg.empty()

    # 영어 본문 암기 버튼 클릭 시 실행
    if st.sidebar.button(":baby_chick: 디처스!! 영어 본문 암기 되와줘", type="primary"):
        file_path = 'data/eng_textbook.txt'
        # 파일을 읽어서 변수에 저장
        with open(file_path, 'r', encoding='utf-8') as file:
            eng_textbook = file.read()
        query = (
            f"디처스야 영어 본문 암기 도와줘. 본문은 아래와 같아. 빈칸 넣기 문제로 객관식 1개와 주관식 1개를 만들어줘.\n\n"
            f"{eng_textbook}"
        )
        st.chat_message("human").write("성적을 부탁해 디처스~~ 본문암기를 도와줘!!")
        with st.spinner('답변중...'):
            pg = st.empty()
            response = do_ask(query, pg, msgs) # ChatGPT에게 물어보는 부분
            st.chat_message("ai").write(response['output'])
            pg.empty()

    # 영어 워드마스터 암기 버튼 클릭 시 실행
    if st.sidebar.button(":penguin: 디처스!! 영어 단어 암기 되와줘", type="primary"):
        file_path = 'data/word_master.txt'
        # 파일을 읽어서 변수에 저장
        with open(file_path, 'r', encoding='utf-8') as file:
            eng_textbook = file.read()
        query = (
            f"디처스야 영어 단어 암기 도와줘. 단어들은 아래와 같아. 이 단어만 암기하면 되. 다른건 물어보지 말아줘. 객관식 2문제만 내줘.\n\n"
            f"{eng_textbook}"
        )
        st.chat_message("human").write("성적을 부탁해 디처스~~ 단어암기를 도와줘!!")
        with st.spinner('답변중...'):
            pg = st.empty()
            response = do_ask(query, pg, msgs) # ChatGPT에게 물어보는 부분
            st.chat_message("ai").write(response['output'])
            pg.empty()

    # 힘이 나는 말 버튼 클릭 시 실행
    if st.sidebar.button(":+1: 디처스!! 힘이 나는 말을 해줘.", type="primary"):
        query = "디처스!! 공부는 왜 해야 하는걸까? 힘이 나는 말을 해줘. 엄마, 아빠, 선생님 처럼 말고. 친구처럼 힘이 나는 말을 해줘."
        st.chat_message("human").write(query)
        with st.spinner('답변중...'):
            pg = st.empty()
            response = do_ask(query, pg, msgs) # ChatGPT에게 물어보는 부분
            st.chat_message("ai").write(response['output'])
            pg.empty()

    if len(msgs.messages) == 0:
        msgs.add_ai_message("나는 너의 공부를 도와주는 디처스야!!")

    # 자유 대화 입력 부분
    user_input = st.chat_input()

    if query := user_input:
        st.chat_message("human").write(query)
        with st.spinner('답변중...'):
            pg = st.empty()
            response = do_ask(query, pg, msgs) # ChatGPT에게 물어보는 부분
            st.chat_message("ai").write(response['output'])
            pg.empty()

def do_ask(user_query, container_name, msgs):
    # 프롬프트 설정(ChatGPT의 역할을 프롬프트 엔지니어링)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "너는 시험준비를 도와주는 친구야. 친구들의 공부를 도와줘. 친구들이 힘들때 응원을 많이 해줘. 너는 누구니라는 질문에 이렇게 답변해줘. 안녕!! 나는 디처스야. 디미고 티쳐스의 줄임 말이야. 디미고 친구들이 재미있게 공부하도록 도와 줄 수 있어. 그 외의 고민 상담도 잘 해줄 수 있어. 우리 재미있게 공부해 보자. 대화 톤은 아주 친한 친구 처럼 얘기해줘. 이모지를 많이 사용해줘."),
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{question}"),
        ]
    )

    # 스트림 핸들러 설정(실시간으로 답변이 보이게 하는 기능)
    st_callback = StreamHandler(container_name)
    # OpenAI 챗봇 모델 설정
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0, streaming=True, callbacks=[st_callback], verbose=True)

    # 검색 툴 설정(타빌리 사용)
    tools = [TavilySearchResults(max_results=1)] #검색엔진으로 타빌리를 사용
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[st_callback], verbose=True, max_iterations=3)

    # 메시지를 기억
    chain_with_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": user_query}, config)

    return response

if __name__ == '__main__':
    main()
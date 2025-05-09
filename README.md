# 부산대학교 정보컴퓨터공학부 2025 전기 졸업과제 

## 1. 프로젝트 소개
### 1.1 프로젝트 주제
 공공데이터를 활용한 KoBERT 파인튜닝과 한국어 키워드 분석 및 대쉬보드 시각화
### 1.2 개발 배경 및 필요성
 자연어 처리 기술의 발전과 함께 텍스트 마이닝 분야에서 키워드 추출의 중요성이 증가하고 있습니다. 기존 오픈소스 키워드 추출 도구는 긴 문장에서 성능 저하와 연산 시간 증가 문제를 보이며, 특히 한국어처럼 형태소 단위로 의미가 구성되는 교착어의 경우 어미나 접사 변화에 따라 단어의 의미가 달라져 키워드를 정밀하게 추출하는 것이 어렵습니다. 예를 들어 KeyBERT는 다양한 사전학습 언어 모델을 활용 가능하지만 키워드 추출 목적에 특화되지 않아 정확도와 일관성이 부족한 한계를 보이고 있습니다. 본 프로젝트는 한국어 특화 언어모델인 KoBERT를 개선하여 이러한 한계를 극복하고자 합니다.
### 1.3 개발 목표 및 주요 내용
 본 프로젝트는 KoBERT를 활용한 BIO 태그 기반 키워드 추출 시스템 개발과 이를 바탕으로 직관적인 대시보드 웹 서비스 구현을 목표로 합니다. 키워드 추출의 정밀성을 높이기 위해, KoBERT 모델을 BIO 태그 방식으로 파인튜닝(fine-tuning)하고, CRF(Conditional Random Field) 레이어를 추가하여 키워드 추출의 연속성과 문맥적 일관성을 강화합니다. 또한 사용자가 보다 직관적이고 효율적으로 텍스트 분석 결과를 활용할 수 있도록, 사용자가 분석을 원하는 텍스트를 입력하거나 이전에 분석한 텍스트를 불러와 확인할 수 있는 대시보드형 웹 서비스를 병행하여 구현하며, 시스템은 입력된 텍스트로부터 추출된 키워드를 기반으로 다양한 시각적 분석 결과를 제공하는 것이 목표입니다.
### 1.4 세부내용
<ul style="list-style-type: disc">
  <li>프론트엔드</li>
  <ul style="list-style-type: circle">
    <li>React를 활용한 사용자 인터페이스 구축</li>
    <li>chart.js를 이용한 데이터 시각화 제공</li>
  </ul>
  <li>백엔드</li>
  <li>자연어처리</li>
</ul>

### 1.5 주차별 계획 및 진행상황


## 2. 시스템 구상도
![image](./src/system_figure.png)

## 3. 멤버
| 박준혁 | 이차현 | 임성표 |
|:-------:|:-------:|:-------:| 
|<a href="https://github.com/JakeFRCSE"><img width="100px" alt="박준혁" src="https://avatars.githubusercontent.com/u/162955476?v=4" /></a>|<a href="https://github.com/chahyunlee"><img width="100px" alt="이차현" src="https://avatars.githubusercontent.com/u/163325051?v=4" /></a>|<a href="https://github.com/LimSungPyo"><img width="100px" alt="임성표" src="https://avatars.githubusercontent.com/u/132332450?v=4" /></a>|
| eppi001004@gmail.com | chahyun20@naver.com | lsp11121@gmail.com |
| 자연어처리 | 프론트엔드 | 백엔드 |

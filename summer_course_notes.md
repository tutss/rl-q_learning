## Curso de verão - IME-USP
## Reinforcement Learning
## Artur Magalhães R dos Santos

### Aula 1
#### 01/02/2021

**Bloco 1**
- vídeo sobre RL da DeepMind
  - AlphaGo, MuZero, etc
- DeepRL tem muitos desafios -> enorme quantidade de dados, engenharia de recompensas, ótimos locais podem ser problemáticos, overfitting, aprendizado instável e resultados difíceis de reproduzir
- área que está engatinhando em resultados de produção

**Bloco 2**
- exemplos de RL -> cartpole e pong
- outras aplicações
- formulação teórica como uma MDP
  - S estados, A ações, p distribuição inicial, P transição e R recompensa
  - o futuro é independente do passado dado o presente -> o presente diz tudo sobre o passado (propriedade de Markov)
  - hoje não é diferente de amanhã (distribuição estacionária)
  - retorno -> soma dos retornos de cada episódio
  - política (estocástica) -> pi
  - função objetivo -> valor esperado da soma das trajetórias

**Bloco 3**
Aprendizado por reforço associado a Deep Learning
- redes feedforwards e convolucionais ao longo do curso
  - aprender a política, função valor ou modelo por meio das redes neurais, que estimam uma função
- generalização, exploração e atribuição de crédito
  - dilema nas escolhas
  - correlacionar ações as recompensas
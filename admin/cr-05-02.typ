#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
title: [Compte Rendu de Réunion 3],
course: [Projet #smallcaps[Ai2d]],
authors: ("Paul Chambaz", "Frédéric Li Combeau"),
university: [Sorbonne Université],
reference: [Master #smallcaps[Ai2d] M1 -- 05 février 2025],
)

== Points clés

=== Revue du travail accompli
- DQN et DDPG ont été implémentés et testés sur Cartpole et MountainCar.
- Les environnements Cartpole continuous et MountainCar continuous ont été étendus avec set_state.
- Continuation de l'étude de AFU.

=== AFU vs Autres Algorithmes
AFU apparaît comme véritablement off-policy, similaire au Q-learning classique, contrairement à DDPG, TD3 et SAC qui sont théoriquement plus proches de SARSA.

=== BBRL
Présentation de l'architecture :

- Basée sur un tableau noir (dictionnaire)
- Agents comme modules PyTorch
- Optimisation pour GPU via tenseurs
- Support pour l'exécution parallèle d'environnements

== Pour la semaine prochaine

- Compléter les trois notebooks _BBRL_.
- Intégrer AFU pytorch.
- Finaliser l'implémentation SAC.
- Produire les premières courbes sur données uniformes.
- Commencer l'adaptation AFU pour BBRL.

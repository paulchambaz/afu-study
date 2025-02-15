#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
  title: [Ordre du jour],
  course: [Projet #smallcaps[Ai2d]],
  authors: ("Paul Chambaz", "Frédéric Li Combeau"),
  university: [Sorbonne Université],
  reference: [Master #smallcaps[Ai2d] M1 -- 05 février 2025],
)

== Points administratifs

Participation et organisation par l'ALIAS de la participation au FOSDEM ce week-end, ce qui a limité le temps de travail disponible cette semaine. Cette situation est exceptionnelle et ne devrait pas se reproduire.

== Travail accompli

- Ajout des environements Cartpole continous et MountainCar continous dans le projet et extension _set_state_. 
- Étude et implémentation de l'algorithme DQN et premières executions sur Cartpole.
- Étude et implémentation de l'algorithme DDPG et premières executions sur Cartpole.
- Première exécution de l'implémentation PyTorch d'AFU sur les environnements de test.
- Lecture du papier complémentaire AFU.

== Objectifs

- Implémenter SAC de la même façon que DQN et DDPG.
- Première évaluation des performances de AFU sur Cartpole continous.
- Première évaluation des performances de SAC sur Cartpole continous et comparaison des performances.
- Établir méthodologie d'évaluation des performances off-policy.

== Points à Discuter

- Retour sur les premières expérimentations avec AFU.
- Validation de l'implémentation de _set_state_ dans les environnements.
- Discussion sur la méthodologie d'évaluation des performances off-policy.

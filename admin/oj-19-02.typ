#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
  title: [Ordre du jour],
  course: [Projet #smallcaps[Ai2d]],
  authors: ("Paul Chambaz", "Frédéric Li Combeau"),
  university: [Sorbonne Université],
  reference: [Master #smallcaps[Ai2d] M1 -- 19 février 2025],
)

== Travail accompli

- Refactorisation de la structure des expériences et simplification des scripts d'évaluation
- Ajout de nouveaux environnements : Pendulum, LunarLander, Ant, Swimmer, BipedalWalker
- Tests comparatifs AFU/DDPG sur BipedalWalker
- Lecture du papier IQL
- Travail pour mieux comprendre l'implémentation d'AFU
- Début de la rédaction du cahier des charges

== Points à discuter

- Point théorique sur IQL
- Point théorique sur AFU
- Point sur le cahier des charges
- Point sur la transition online/offline -- pas de points avant deux semaines

== Objectifs

- Finalisation du cahier des charges
- Correction et uniformisation des environnements
- Analyse comparative des expériences on-policy et off-policy

#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
  title: [Ordre du jour],
  course: [Projet #smallcaps[Ai2d]],
  authors: ("Paul Chambaz", "Frédéric Li Combeau"),
  university: [Sorbonne Université],
  reference: [Master #smallcaps[Ai2d] M1 -- 29 janvier 2025],
)

== Revue du travail accompli

L'infrastructure technique est maintenant en place avec un repository _git_
fonctionnel incluant les hooks de pré-commit pour _black_ et _pylint_, ainsi
qu'une pipeline d'intégration continue _Jenkins_. Nous avons également réalisé
les premières implémentations tests avec une version préliminaire de _SAC_, un
environnement _CartPole_ et un cycle d'entraînement basique. L'analyse initiale
de l'article _AFU_ est terminée et nous avons commencé à nous familiariser avec
_BBRL_.

== Objectifs de la Semaine

Notre priorité est d'implémenter _AFU alpha_ en suivant la même méthodologie
que celle utilisée pour _SAC_, en nous appuyant sur les ressources fournies.
Parallèlement, nous devons établir une méthodologie d'évaluation rigoureuse
pour comparer les performances des algorithmes. Cette méthodologie sera
implémentée dans un script _Python_ qui s'intégrera à notre pipeline _Jenkins_,
permettant des évaluations automatisées et reproductibles. L'objectif est
d'avoir un système robuste pour commencer les tests de comparaison, en
particulier pour l'évaluation des capacités _off-policy_ d'AFU.

== Points à Discuter

Nous devons faire un point technique approfondi sur AFU pour valider notre
compréhension collective de l'algorithme. Une discussion est nécessaire
concernant la modification de _Gymnasium_ pour le tirage aléatoire dans l'espace
d'état : faut-il opter pour une extension de la bibliothèque ou une
modification directe du code source ?

Enfin pour la semaine du 5 au 12 février, nous devrons clarifier et consolider
notre implémentation actuelle en une pipeline cohérente et robuste avant de
commencer les tests comparatifs sur les propriétés off-policy, qui constituent
notre premier objectif de résultats.

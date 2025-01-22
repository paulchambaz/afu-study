#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
  title: [Compte Rendu de Réunion 1],
  course: [Projet #smallcaps[Ai2d]],
  authors: ("Paul Chambaz", "Frédéric Li Combeau"),
  university: [ Sorbonne Université ],
  reference: [ Master #smallcaps[Ai2d] M1 -- 22 janvier 2025 ],
)

== Présentation du Projet

Le projet s'articule autour de l'étude de l'algorithme AFU (_Actor-Free Updates_), récemment proposé par Nicolas Perrin. Bien que l'article original soit mathématiquement complexe, l'algorithme présente un potentiel significatif qui mérite d'être exploré et mis en valeur.

== Objectif Principal

L'objectif principal est d'étudier les propriétés _off-policy_ d'AFU. L'hypothèse est qu'AFU pourrait être véritablement off-policy, contrairement à de nombreux algorithmes de deep RL actuels qui, bien que classés comme off-policy, montrent des limitations significatives avec des données éloignées de leur politique courante. Pour tester cette hypothèse, il faut utiliser des données générées uniformément dans l'espace état-action, un scénario où le Q-Learning classique converge mais où la plupart des algorithmes de deep RL échouent.

Si le temps le permet, il est possible également explorer un second axe : la capacité d'AFU à gérer efficacement la transition entre apprentissage _offline_ et _online_, où la plupart des algorithmes actuels montrent une dégradation temporaire des performances.

== Approche Méthodologique

L'étude débutera avec des environnements simples de _Gymnasium_, notamment _CartPole_ en version continue, _MountainCar_ ou _Pendulum_. Ces environnements permettront une première validation des propriétés d'AFU. La difficulté technique principale résidera dans la génération d'états uniformes, l'état représentant l'interaction entre l'agent et l'environnement. Une modification de l'API des environnements sera nécessaire pour permettre la configuration directe des états.

L'implémentation se fera dans le framework BBRL, qui offre une base solide pour le développement d'algorithmes de RL.

== Aspects Administratifs et Organisation

Le projet s'organise autour de réunions hebdomadaires, avec possibilité de passage à un rythme bihebdomadaire à mi-semestre selon les besoins. Chaque réunion nécessite l'envoi d'un ordre du jour 24h à l'avance, le mardi matin et est suivie d'un compte-rendu.

Les livrables attendus comprennent :
- Un cahier des charges à mi-semestre
- Un travail bibliographique documenté
- Un rapport final
- Une soutenance devant un jury, incluant M. Sigaud

L'évaluation sera collective et prendra en compte l'avis de l'encadrant. La note finale sera basée sur l'ensemble des livrables et la présentation orale.

== Prochaine Étape

La priorité est de se familiariser avec les concepts fondamentaux du RL et du deep RL, en commençant par le cours sur le RL tabulaire prévu pour vendredi et BBRL. Le mise en place du repository Github, des libraries et d'un environement de test automatisé est aussi prévue pour cette semaine.

#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
title: [Compte Rendu de Réunion 5],
course: [Projet #smallcaps[Ai2d]],
authors: ("Paul Chambaz", "Frédéric Li Combeau"),
university: [Sorbonne Université],
reference: [Master #smallcaps[Ai2d] M1 -- 19 février 2025],
)

== Points clés

=== Apprentissage Offline/Online

- Discussion sur les problématiques spécifiques : biais de surestimation et changement de distribution
- Revue des différentes approches existantes (CQL, CAL-QL, IQL) et leurs limitations
- Avantages potentiels d'AFU pour la transition offline/online

=== Aspects Techniques

- Discussion sur l'utilisation de d4RL et Antmaze
- Explication de la expectile regression d'AFU
- Point sur l'implémentation d'AFU
- Questions sur l'uniformité de l'échantillonnage des états
- Limite de set_state sur l'apprentissage en pratique

== Axes de travail

- Axe 1 : Finalisation de l'implémentation AFU dans BBRL
- Axe 2 : Expériences sur Pendulum et optimisation des hyperparamètres
- Axe 3 : Mise en place des expériences de transition offline/online

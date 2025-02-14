#import "template.typ": *
#import "@preview/cetz:0.3.1": canvas, draw

#show: report.with(
title: [Compte Rendu de Réunion 4],
course: [Projet #smallcaps[Ai2d]],
authors: ("Paul Chambaz", "Frédéric Li Combeau"),
university: [Sorbonne Université],
reference: [Master #smallcaps[Ai2d] M1 -- 12 février 2025],
)

== Points clés

=== BBRL et méthodologie
- Complétion des notebooks BBRL, intérêt des multiples environements gymnasium pour la suite
- Hyperparamètres : phase de compréhension manuelle, puis phase de comparaison avec budge équitable

=== Revue des implémentations
- SAC avec BBRL : revue de la théorie en détail et revue de l'implémentation
- AFU avec BBRL : performances qui semblent inférieures à la version de M. Perrin, poursuite du développement.

=== Résultats expérimentaux
- Implémentation AFU sous-optimale
- Convergent inattendue des trois algorithmes en off-policy
- Stabilité post-convergence en off-policy

== Pour la semaine prochaine

=== Technique
- Tests avec code de M. Perrin
- Extension de set_state (pendulum)
- Vérification uniformité échantillonnage
- Amélioration implémentation AFU
- Étude IQL

=== Organisation
- Cahier des charges
- Communication résultats intermédiaires

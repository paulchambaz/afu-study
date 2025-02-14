#import "@preview/cetz:0.3.1": canvas, draw
#import "@preview/based:0.2.0": base64

#let report(
  title: none,
  course: none,
  authors: (),
  university: none,
  reference: none,
  bibliography-path: "",
  nb-columns: 1,
  doc
) = {
  set text(size: 11pt, lang: "fr", font: "New Computer Modern")

  show math.equation: set block(breakable: true)

  set enum(numbering: "1. a.")
  set list(marker: [--])

  set page(
    numbering: "1",
    margin: (x: 2cm, y: 3cm),
    header: [
      #set text(weight: 400, size: 10pt)
      #stack(dir: ttb, 
        stack(dir: ltr,
          course,
          h(1fr),
          [ #authors.join(" ", last: " & ") ],
        ),
        v(.1cm),
        line(length: 100%, stroke: .4pt)
      )
    ],
    footer: [
      #set text(weight: 400, size: 10pt)
      #stack(dir: ltr,
          university,
          h(1fr),
          [ #context { counter(page).display("1") } ],
          h(.7fr),
          reference,
      )
    ],
  )

  set par(justify: true)

  show heading.where(
    level: 2
  ): it => block(width: 100%)[
    #v(0.2cm)
    #set align(center)
    #set text(13pt, weight: 500)
    #smallcaps(it.body)
    #v(0.2cm)
  ]

  show heading.where(
    level: 3
  ): it => text(
    size: 11pt,
    weight: "regular",
    style: "italic",
    it.body + [.],
  )

  show heading.where(
    level: 4
  ): it => text(
    size: 11pt,
    weight: "regular",
    style: "italic",
    h(1em) + [(] + it.body + [)],
  )

  align(center)[
    #v(.5cm)
    #rect(inset: .4cm, stroke: .4pt)[
      = #title
    ]
    #v(1cm)
  ]

  if nb-columns > 1 {
    show: rest => columns(nb-columns, rest)
    doc

    if bibliography-path != "" {
      bibliography(title: [ == Bibliographie ], bibliography-path, style: "association-for-computing-machinery")
    }
  } else {
    doc

    if bibliography-path != "" {
      bibliography(title: [ == Bibliographie ], bibliography-path, style: "association-for-computing-machinery")
    }
  }

}

#let hidden-bib(body) = {
  box(width: 0pt, height: 0pt, hide(body))
}

#let node(pos, key, value: none, radius: 0.4cm) = {
  draw.circle(pos, radius: radius, name: key)

  if value == none {
    draw.content(key, key)
  } else {
    draw.content(key, value)
  }
}

#let edge(start, end, pos: 50%, anchor: "south", orientation: "north", value: none) = {
  draw.line(start, end, name: start + "_" + end)

  if orientation == "follow" {
    draw.content((start + "_" + end + ".start", pos, start + "_" + end + ".end"), angle: start + "_" + end + ".end", padding: .1cm, anchor: anchor, value)
  } else {
    draw.content((start + "_" + end + ".start", pos, start + "_" + end + ".end"), padding: .1cm, anchor: anchor, value)
  }
}

#let arc(start, end, pos: 50%, anchor: "south", orientation: "north", value: none) = {
  draw.set-style(
    mark: (fill: black),
  )
  draw.line(start, end, name: start + "_" + end, mark: (end: "stealth"))

  if orientation == "follow" {
    draw.content((start + "_" + end + ".start", pos, start + "_" + end + ".end"), angle: start + "_" + end + ".end", padding: .1cm, anchor: anchor, value)
  } else {
    draw.content((start + "_" + end + ".start", pos, start + "_" + end + ".end"), padding: .1cm, anchor: anchor, value)
  }
}

#let proof(body) = {
  box(width: 100%, stroke: 1pt + black, inset: 1em)[
    #emph[Démonstration.]
    #body

    #align(end)[$qed$]
  ]
}

#let answer(body) = {
  block(breakable: true, width: 100%, stroke: 1pt + black, inset: 1em)[
    #emph[Réponse.]
    #body
    #align(end)[$qed$]
  ]
}

#let notes(body) = {
  block(breakable: true, width: 100%, stroke: (dash: "dashed"), inset: 1em)[
    #emph[Théorie.]
    #body
  ]
}


#let random = { math.attach(sym.arrow.l.long, t: [
  #box(stroke: black + .5pt, width: 0.6em, height: 0.6em, inset: 0.06em)[
    #grid(columns: (0.16em, 0.16em, 0.16em), rows: (0.16em, 0.16em, 0.16em),
      [], [], align(center + horizon, circle(radius: .06em, fill: black)), [], align(center + horizon, circle(radius: .06em, fill: black)), [], align(center + horizon, circle(radius: .06em, fill: black)), [], [],
    )
  ]
  #box(stroke: black + .5pt, width: 0.6em, height: 0.6em, inset: 0.06em)[
    #grid(columns: (0.16em, 0.16em, 0.16em), rows: (0.16em, 0.16em, 0.16em),
      align(center + horizon, circle(radius: .06em, fill: black)), [], align(center + horizon, circle(radius: .06em, fill: black)), [], [], [], align(center + horizon, circle(radius: .06em, fill: black)), [], align(center + horizon, circle(radius: .06em, fill: black)),
    )
  ]
]) }

#let algorithm(title: none, input: none, output: none, steps: ()) = {
  figure(canvas(length: 100%, {
    import draw: *

    if title != none {
      line((0, 0), (1, 0))
      content((1em, -.3em), anchor: "north-west", title)
      line((0, -1.4em), (1, -1.4em))
    }

    if input != none {
      content((2em, -1.8em), anchor: "north-west", [ *Entrée* : #input ])
    }
    if output != none {
      content((2em, -3.0em), anchor: "north-west", [ *Sortie* : #output ])
    }

    for (i, step) in steps.enumerate() {


      content(
        (1em, -4.2em - i * 1.2em),
        anchor: "north-east",
        text(size: 0.8em, weight: 700)[ #(i + 1) ],
      )

      if type(step) != dictionary {
        step = (depth: 0, line: step)
      }

      content(
        (2em + step.depth * 1em, -4.2em - i * 1.2em),
        anchor: "north-west",
        step.line
      )
  }

  }))
}

#let bar(value) = math.accent(value, "-")


#let FT = { "FT" }
#let si = { "si" }
#let sinon = { "sinon" }
#let DFT = { "DFT" }
#let Rect = { "Rect" }
#let integ = { $integral_(-oo)^(+oo)$ }
#let sumk(k) = { $sum_(#k=-oo)^(+oo)$ }

#let comb = {
  rect(stroke: none, width: 0.6cm, height: 0.4cm, inset: 2pt, outset: 0pt)[
    #place[#line(start: (0%, 100%), end: (100%, 100%))]
    #place[#line(start: (0%, 0%), end: (0%, 100%))]
    #place[#line(start: (100%, 0%), end: (100%, 100%))]
    #place[#line(start: (50%, 0%), end: (50%, 100%))]
  ]
}

#let scr(it) = text(
  features: ("ss01",),
  box($cal(it)$),
)

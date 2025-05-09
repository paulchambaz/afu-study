#import "@preview/cetz:0.3.1": canvas, draw, coordinate, util, intersection, vector
#import "@preview/based:0.2.0": base64

#let report(
  title: none,
  course: none,
  authors: (),
  university: none,
  reference: none,
  bibliography-path: "",
  nb-columns: 1,
  abstract: none,
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
          [ #authors.join(", ", last: " & ") ],
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
          h(.95fr),
          reference,
      )
    ],
  )

  set par(justify: true)

  align(center)[
    #v(.5cm)
    #rect(inset: .4cm, stroke: .4pt)[
      = #title
    ]
    #v(0.6cm)
  ]

  if abstract != none {
    align(center, text(13pt, weight: 700, smallcaps[Abstract]))
    pad(x: 1cm, bottom: 0.5cm, abstract)
  }

  show heading.where(
    level: 2
  ): it => block(width: 100%)[
    #v(0.2cm)
    #set align(center)
    #set text(13pt, weight: 700)
    #pad(x: 1cm, smallcaps(it.body))
    #v(0.2cm)
  ]

  show heading.where(
    level: 3
  ): it => text(
    size: 11pt,
    weight: 700,
    style: "italic",
    it.body + [.],
  )

  show heading.where(
    level: 4
  ): it => text(
    size: 11pt,
    weight: 700,
    style: "italic",
    h(1em) + [(] + it.body + [)],
  )

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

#let point(pos, key, value: none, radius: 0.4cm) = {
  draw.circle(pos, radius: 0.05cm, fill: black, stroke: none, name: key)

  if value == none {
    draw.content(key, key)
  } else {
    draw.content(key, value)
  }
}

#let node(pos, key, value: none, radius: 0.4cm, kind: "circle") = {
  draw.get-ctx(ctx => {
    let ctx-length = ctx.length.cm()

    if kind == "circle" {
      draw.circle(pos, radius: radius, name: key, fill: white)
    } else if kind == "square" {
      let x = pos.at(0)
      let y = pos.at(1)
      let radius = if type(radius) == length { radius.cm() / ctx-length } else { radius }
        draw.rect((x - radius, y - radius), (x + radius, y + radius), name: key)
    }

    if value == none {
      draw.content(key, key)
    } else {
      draw.content(key, value)
    }
  })
}

#let edge(start, end, pos: 50%, anchor: "south", orientation: "north", value: none) = {
  draw.line(start, end, name: start + "_" + end)

  if orientation == "follow" {
    draw.content((start + "_" + end + ".start", pos, start + "_" + end + ".end"), angle: start + "_" + end + ".end", padding: .1cm, anchor: anchor, value)
  } else {
    draw.content((start + "_" + end + ".start", pos, start + "_" + end + ".end"), padding: .1cm, anchor: anchor, value)
  }
}

#let arc(start, end, pos: 50%, anchor: "south", orientation: "north", value: none, alpha: 0) = {
  draw.set-style(
    mark: (fill: black),
  )
  draw.get-ctx(ctx => {
    // Helper function based on the original implementation
    let element-line-intersection(ctx, elem, a, b) = {
      // Apply transformation to the line points
      let (ta, tb) = util.apply-transform(ctx.transform, a, b)
      
      let pts = ()
      for drawable in elem.at("drawables", default: ()).filter(d => d.type == "path") {
        pts += intersection.line-path(ta, tb, drawable)
      }
      
      return if pts == () {
        a
      } else {
        // Find the nearest point to b
        let pt = util.sort-points-by-distance(tb, pts).first()
        // Reverse the transformation
        return util.revert-transform(ctx.transform, pt)
      }
    }
    
    // Resolve coordinates and determine if they are elements
    let start_system = coordinate.resolve-system(start)
    let end_system = coordinate.resolve-system(end)
    let (ctx, co_start, co_end) = coordinate.resolve(ctx, start, end)
    
    // Check if we're creating a self-loop (start == end)
    let is_self_loop = start == end

    if is_self_loop {
      // For a self-loop, we'll create a curved arc that loops back to the same node
      
      // Get the center point of the element
      let center = co_start
      
      // Use alpha to scale the loop size
      // Base offset for the control points - this will be multiplied by alpha
      let base_offset = 0.75
      
      // The actual offset used will range from a small value when alpha is 0
      // to the full base_offset when alpha is 1
      let min_offset = 0.2  // Minimum offset for visibility when alpha is 0
      let offset = min_offset + (base_offset - min_offset) * alpha
      
      // Calculate control point positions based on anchor direction
      let (ctrl1, ctrl2) = if anchor == "south" or anchor == "bottom" {
        // Loop above the node
        ((center.at(0) - offset, center.at(1) - offset), 
         (center.at(0) + offset, center.at(1) - offset))
      } else if anchor == "north" or anchor == "top" {
        // Loop below the node
        ((center.at(0) - offset, center.at(1) + offset), 
         (center.at(0) + offset, center.at(1) + offset))
      } else if anchor == "east" or anchor == "right" {
        // Loop to the right of the node
        ((center.at(0) + offset, center.at(1) - offset), 
         (center.at(0) + offset, center.at(1) + offset))
      } else if anchor == "west" or anchor == "left" {
        // Loop to the left of the node
        ((center.at(0) - offset, center.at(1) - offset), 
         (center.at(0) - offset, center.at(1) + offset))
      } else if anchor == "south-west" {
        // Loop to the top-right of the node
        ((center.at(0) + 0.5*offset, center.at(1) - offset), 
         (center.at(0) + offset, center.at(1) - 0.5*offset))
      } else if anchor == "south-east" {
        // Loop to the top-left of the node
        ((center.at(0) - offset, center.at(1) - 0.5*offset), 
         (center.at(0) - 0.5*offset, center.at(1) - offset))
      } else if anchor == "north-west" {
        // Loop to the bottom-right of the node
        ((center.at(0) + offset, center.at(1) + 0.5*offset), 
         (center.at(0) + 0.5*offset, center.at(1) + offset))
      } else if anchor == "north-east" {
        // Loop to the bottom-left of the node
        ((center.at(0) - 0.5*offset, center.at(1) + offset), 
         (center.at(0) - offset, center.at(1) + 0.5*offset))
      } else {
        // Default to east if anchor is not recognized
        ((center.at(0) + offset, center.at(1) - offset), 
         (center.at(0) + offset, center.at(1) + offset))
      }
      
      // Find the intersection points with the element's border
      if start_system == "element" {
        let elem = ctx.nodes.at(start)
        
        // Find exit point (where the curve leaves the node)
        let exit_point = element-line-intersection(ctx, elem, center, ctrl1)
        
        // Find re-entry point (where the curve returns to the node)
        let entry_point = element-line-intersection(ctx, elem, center, ctrl2)
        
        // Draw the bezier with the corrected points
        draw.bezier(exit_point, entry_point, ctrl1, ctrl2, name: start + "_" + end, mark: (end: "stealth"))
        
        // Position the label at the middle of the loop
        let label_position = vector.lerp(ctrl1, ctrl2, 0.5)
        
        // Determine a suitable label anchor position (opposite to the loop direction)
        let label_anchor = if anchor.contains("north") {
          "south"  // If loop is above, label should be anchored at its bottom
        } else if anchor.contains("south") {
          "north"  // If loop is below, label should be anchored at its top
        } else if anchor.contains("east") {
          "west"   // If loop is to the right, label should be anchored at its left
        } else if anchor.contains("west") {
          "east"   // If loop is to the left, label should be anchored at its right
        } else {
          "center" // Default
        }
        
        // Position the label
        draw.content(label_position, anchor: label_anchor, value)
      }
    } else {
      // Original case for different start and end points
      // Calculate control points for the bezier curve
      let x1 = co_start.at(0)
      let y1 = co_start.at(1)
      let x2 = co_end.at(0)
      let y2 = co_end.at(1)
      let Ax = (x1 + x2 - y1 + y2) / 2
      let Ay = (y1 + y2 - x2 + x1) / 2
      let Bx = (x1 + x2) / 2
      let By = (y1 + y2) / 2
      let Hx = alpha * Ax + (1 - alpha) * Bx
      let Hy = alpha * Ay + (1 - alpha) * By
      let helper = (Hx, Hy)
      
      // Find border intersections if needed
      if start_system == "element" {
        let elem = ctx.nodes.at(start)
        // For start, we want the intersection in the direction of helper
        co_start = element-line-intersection(ctx, elem, co_start, helper)
      }
      
      if end_system == "element" {
        let elem = ctx.nodes.at(end)
        // For end, we want the intersection in the direction of helper
        co_end = element-line-intersection(ctx, elem, co_end, helper)
      }
      
      // Now draw the bezier curve with the corrected points
      draw.bezier(co_start, co_end, helper, helper, name: start + "_" + end, mark: (end: "stealth"))
      
      if orientation == "follow" {
        draw.content((start + "_" + end + ".start", pos, start + "_" + end + ".end"), 
                      angle: start + "_" + end + ".end", 
                      padding: .1cm, 
                      anchor: anchor, 
                      value)
      } else {
        draw.content(helper, padding: .1cm, anchor: anchor, value)
      }
    }
  })
}

#let proof(body) = {
  box(width: 100%, stroke: 1pt + black, inset: 1em)[
    #emph[Démonstration.]
    #body

    #align(end)[$qed$]
  ]
}

#let answer(hidden: false, body) = {
    block(breakable: true, width: 100%, stroke: 1pt + black, inset: 1em)[
      #emph[Réponse.]
      #if hidden == true {
        hide(body)
      } else {
        body
      }
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
  canvas(length: 100%, {
    import draw: *

    if title != none {
      line((0, 0), (1, 0))
      content((1em, -.3em), anchor: "north-west", title)
      line((0, -1.4em), (1, -1.4em))
    }

    if input != none {
      content((2em, -1.8em), anchor: "north-west", [ *Input*: #input ])
    }
    if output != none {
      content((2em, -3.0em), anchor: "north-west", [ *Ouput*: #output ])
    }

    for (i, step) in steps.enumerate() {
      content(
        (1em, -4.4em - i * 1.2em),
        anchor: "north-east",
        text(size: 0.8em, weight: 700)[ #(i + 1) ],
      )

      if type(step) != dictionary {
        step = (depth: 0, line: step)
      }

      content(
        (2em + step.depth * 1em, -4.4em - i * 1.2em),
        anchor: "north-west",
        step.line
      )
  }

  })
}

#let bar(value) = math.accent(value, "-")

#let OPT = { "OPT" }
#let FT = { "FT" }
#let si = { "si" }
#let sinon = { "sinon" }
#let et = { "et" }
#let ou = { "ou" }
#let non = { "non" }
#let tq = { "t.q." }
#let avec = { "avec" }
#let DFT = { "DFT" }
#let Rect = { "Rect" }
#let argmin = { "argmin" }
#let argmax = { "argmax" }
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
#let rel = math.class(
  "relation",
  [ #math.cal("R") ]
)

#let calc-width(left: none, right: none) = {
  import draw: *
  if left == none and right == none { return 1.0 }

  let left-width = if left != none { calc-width(
    left: left.at("left", default: none),
    right: left.at("right", default: none)
  )} else { 0.0 }

  let right-width = if right != none { calc-width(
    left: right.at("left", default: none),
    right: right.at("right", default: none)
  )} else { 0.0 }

  return calc.max(left-width + right-width, 1.0)
}
    
#let min-separation(depth) = {
  import draw: *
  return calc.max(2.0 * calc.pow(0.8, depth), 1.0)
}
    
#let get-offset(left-width, right-width, depth, spread) = {
  import draw: *
  let min-sep = min-separation(depth)

  let total-width = left-width + right-width
  let base-offset = calc.max(total-width * 0.5, min-sep) * spread

  return base-offset * calc.pow(0.8, depth)
}
    
#let bintree(position, label, left: none, right: none, depth: 0, spread: 0.6) = {
  import draw: *
  let left-width = if left != none { 
    calc-width(
      left: left.at("left", default: none),
      right: left.at("right", default: none)
    )
  } else { 0.0 }

  let right-width = if right != none {
    calc-width(
      left: right.at("left", default: none),
      right: right.at("right", default: none)
    )
  } else { 0.0 }

  let x-offset = get-offset(left-width, right-width, depth, spread)

  if left != none {
  let left-pos = (position.at(0) - x-offset, position.at(1) - 1)
  line(position, left-pos)
  bintree(
    left-pos,
    left.label,
    left: left.at("left", default: none),
    right: left.at("right", default: none),
    depth: depth + 1,
    spread: spread
  )
}

  if right != none {
  let right-pos = (position.at(0) + x-offset, position.at(1) - 1)
  line(position, right-pos)
  bintree(
    right-pos,
    right.label,
    left: right.at("left", default: none),
    right: right.at("right", default: none),
    depth: depth + 1,
    spread: spread
  )
}

  circle(position, radius: 0.4cm, fill: white)
  content((position.at(0), position.at(1) + 0.0), label)
}

#let cube(x, y, z, value: none) = {
  let len = 0.2
  draw.line(
    (
    x * len + y * len * 0.5,
    y * len * 0.5 + z * len
  ),
    (
    (x + 1) * len + y * len * 0.5,
    y * len * 0.5 + z * len
  ),
    (
    (x + 1) * len + y * len * 0.5,
    len + y * len * 0.5 + z * len
  ),
    (
    x * len + y * len * 0.5,
    len + y * len * 0.5 + z * len
  ),
    close: true, fill: gray.lighten(60%), stroke: (paint: gray.lighten(20%), thickness: 0.5pt, cap: "round", join: "round"),
  )

  draw.line(
    (
    (x + 1) * len + y * len * 0.5,
    y * len * 0.5 + z * len
  ),
    (
    (x + 1) * len + y * len * 0.5,
    len + y * len * 0.5 + z * len
  ),
    (
    (x + 1) * len + (y + 1) * len * 0.5,
    len + (y + 1) * len * 0.5 + z * len
  ),
    (
    (x + 1) * len + (y + 1) * len * 0.5,
    (y + 1) * len * 0.5 + z * len
  ),
    close: true, fill: gray.lighten(40%), stroke: (paint: gray.lighten(20%), thickness: 0.5pt, cap: "round", join: "round"),
  )

  draw.line(
    (
    x * len + y * len * 0.5,
    len + y * len * 0.5 + z * len
  ),
    (
    (x + 1) * len + y * len * 0.5,
    len + y * len * 0.5 + z * len
  ),
    (
    (x + 1) * len + (y + 1) * len * 0.5,
    len + (y + 1) * len * 0.5 + z * len
  ),
    (
    x * len + (y + 1) * len * 0.5,
    len + (y + 1) * len * 0.5 + z * len
  ),
    close: true, fill: gray.lighten(80%), stroke: (paint: gray.lighten(20%), thickness: 0.5pt, cap: "round", join: "round"),
  )

  draw.content(((x + 0.5) * len + y * len * 0.5, y * len * 0.5 + (z + 0.5) * len), value)
}


#let indifference(a, b, alpha: 0.5, stroke: red) = {
  let ax = a.at(0)
  let ay = a.at(1)

  let bx = b.at(0)
  let by = b.at(1)

  let cx = (ax + bx) / 2
  let cy = (ay + by) / 2

  let dx = calc.min(ax, bx)
  let dy = calc.min(ay, by)

  let ex = (1 - alpha) * cx + alpha * dx
  let ey = (1 - alpha) * cy + alpha * dy

  draw.bezier(a, b, (ex, ey), stroke: stroke)
  draw.circle(a, radius: 1pt, fill: black)
  draw.circle(b, radius: 1pt, fill: black)
}

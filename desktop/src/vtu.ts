// Minimal VTU (ParaView XML UnstructuredGrid) ASCII parser.
// Sufficient for files written by meshio with format="ascii".
// Limitations: does not handle base64/appended/compressed encodings.

export type VtuMesh = {
  points: number[];                  // flat [x0,y0,z0, x1,y1,z1, ...]
  cells: number[][];                 // each row is the connectivity of one cell
  cellTypes: number[];               // VTK cell type per cell (10=tetra, 12=hex)
  scalars: Record<string, number[]>; // per-point scalar arrays
};

export function parseVtu(xml: string): VtuMesh {
  const doc = new DOMParser().parseFromString(xml, "text/xml");
  const piece = doc.querySelector("Piece");
  if (!piece) throw new Error("VTU: no <Piece>");

  const points = readArray(piece.querySelector("Points > DataArray"));
  const conn = readIntArray(
    findDataArray(piece, "Cells", "connectivity"),
  );
  const offs = readIntArray(findDataArray(piece, "Cells", "offsets"));
  const types = readIntArray(findDataArray(piece, "Cells", "types"));

  const cells: number[][] = [];
  let last = 0;
  for (const o of offs) {
    cells.push(Array.from(conn.slice(last, o)));
    last = o;
  }

  const scalars: Record<string, number[]> = {};
  const pd = piece.querySelector("PointData");
  if (pd) {
    pd.querySelectorAll("DataArray").forEach((el) => {
      const name = el.getAttribute("Name") ?? "";
      if (!name) return;
      const nc = parseInt(el.getAttribute("NumberOfComponents") ?? "1");
      const a = readArray(el);
      // Only keep scalar fields for now (NumberOfComponents = 1)
      if (nc === 1) scalars[name] = a;
    });
  }

  return { points, cells, cellTypes: types, scalars };
}

function findDataArray(
  piece: Element,
  parent: string,
  name: string,
): Element | null {
  const items = piece.querySelectorAll(`${parent} > DataArray`);
  for (const el of Array.from(items)) {
    if (el.getAttribute("Name") === name) return el;
  }
  return null;
}

function readArray(el: Element | null): number[] {
  if (!el) return [];
  const fmt = el.getAttribute("format") ?? "ascii";
  if (fmt !== "ascii") {
    throw new Error(
      `VTU DataArray format="${fmt}" not supported (ASCII only)`,
    );
  }
  const text = (el.textContent ?? "").trim();
  if (!text) return [];
  // Split on any whitespace; cheap parseFloat
  const toks = text.split(/\s+/);
  const out = new Array<number>(toks.length);
  for (let i = 0; i < toks.length; i++) out[i] = parseFloat(toks[i]);
  return out;
}

function readIntArray(el: Element | null): number[] {
  return readArray(el).map((v) => v | 0);
}

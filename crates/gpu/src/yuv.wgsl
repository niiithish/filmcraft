// Export output for 8-bit 4:2:0 encoders: the linear premultiplied accumulator over black, encoded
// to 8-bit sRGB codes through the CPU's own table (`filmcraft_color::linear_to_srgb_u8_table`),
// then BT.709 limited-range Y'CbCr with a 2×2 chroma average, in the order of operations of
// `filmcraft_export::rgba_to_yuv420_8` (per-code tables for `c / 255` and `k · c / 255`).
//
// One invocation converts 8 pixels × 2 rows and writes whole words: two Y words per row, one U
// word and one V word (the width is a multiple of 8, the height even). Output, as u32 words of
// little-endian bytes: Y (w·h), then U and V ((w/2)·(h/2) each).

struct Params {
    w: u32,
    h: u32,
    cw: u32,
    ch: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
// [0, 4096): sRGB code of linear index i/4095; then 256 each of c/255, 0.2126·c/255,
// 0.7152·c/255 and 0.0722·c/255 as f32 bits
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

const UNIT: u32 = 4096u;
const KR: u32 = 4352u;
const KG: u32 = 4608u;
const KB: u32 = 4864u;

fn srgb(v: f32) -> u32 {
    let i = u32(clamp(v, 0.0, 1.0) * 4095.0 + 0.5);
    return lut[min(i, 4095u)];
}

fn tab(base: u32, c: u32) -> f32 {
    return bitcast<f32>(lut[base + min(c, 255u)]);
}

// `filmcraft_export`'s code8: round half up, clamped to 1..254
fn code8(x: f32) -> u32 {
    return u32(clamp(i32(x + 0.5), 1, 254));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x0 = id.x * 8u;
    let y0 = id.y * 2u;
    if (x0 >= p.w || y0 >= p.h) {
        return;
    }
    var us = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var vs = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var r = 0u; r < 2u; r++) {
        let y = y0 + r;
        var lo = 0u;
        var hi = 0u;
        for (var k = 0u; k < 8u; k++) {
            let px = textureLoad(src, vec2<i32>(i32(x0 + k), i32(y)), 0);
            let cr = srgb(px.r);
            let cg = srgb(px.g);
            let cb = srgb(px.b);
            let yy = tab(KR, cr) + tab(KG, cg) + tab(KB, cb);
            let code = code8(16.0 + 219.0 * yy);
            if (k < 4u) {
                lo = lo | (code << (8u * k));
            } else {
                hi = hi | (code << (8u * (k - 4u)));
            }
            let c = k / 2u;
            us[c] = us[c] + (tab(UNIT, cb) - yy) / 1.8556;
            vs[c] = vs[c] + (tab(UNIT, cr) - yy) / 1.5748;
        }
        let at = (y * p.w + x0) / 4u;
        out[at] = lo;
        out[at + 1u] = hi;
    }
    var uw = 0u;
    var vw = 0u;
    for (var c = 0u; c < 4u; c++) {
        uw = uw | (code8(128.0 + 224.0 * us[c] / 4.0) << (8u * c));
        vw = vw | (code8(128.0 + 224.0 * vs[c] / 4.0) << (8u * c));
    }
    let chroma = (id.y * p.cw + id.x * 4u) / 4u;
    let u_at = (p.w * p.h) / 4u + chroma;
    let v_at = (p.w * p.h + p.cw * p.ch) / 4u + chroma;
    out[u_at] = uw;
    out[v_at] = vw;
}

// Resizing a picture exactly the way Pillow does it.
//
// Training prepared every picture with torchvision's
// transforms.Resize(256, BILINEAR) followed by CenterCrop(224), on a PIL image.
// A browser canvas has its own, different, resizing filter. This file does the
// arithmetic of Pillow's resize (src/libImaging/Resample.c) by hand instead, so
// that the page and training turn the same picture into the same numbers:
//
//   * a bilinear "tent" filter whose width grows with the shrink factor, so a
//     big photo is averaged (anti-aliased) and not sampled
//   * the weights are made whole numbers with 22 bits of fraction
//   * the horizontal pass runs first and is rounded to 8 bits, and then the
//     vertical pass runs on that
//
// It also copies two small decisions of torchvision: how the new size is
// worked out, and how a half pixel is rounded when cutting out the middle.

(function () {
  const PRECISION_BITS = 32 - 8 - 2;  // Pillow's fixed point, 22 bits of fraction

  // Pillow's bilinear filter, with support 1.0
  function bilinear(x) {
    if (x < 0) x = -x;
    return x < 1 ? 1 - x : 0;
  }

  // For every output position: which input pixels it reads (first, count) and
  // the whole-number weight of each. Same as Pillow's precompute_coeffs
  // followed by normalize_coeffs_8bpc.
  function coefficients(inSize, outSize) {
    const scale = inSize / outSize;
    const filterScale = Math.max(scale, 1);
    const support = filterScale;  // the filter's support (1.0) times the stretch
    const ksize = Math.ceil(support) * 2 + 1;
    const ss = 1 / filterScale;

    const bounds = new Int32Array(outSize * 2);
    const weights = new Int32Array(outSize * ksize);
    const real = new Float64Array(ksize);

    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      // C casts to int, which cuts off the fraction (toward zero).
      let xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;

      let total = 0;
      for (let x = 0; x < xmax; x++) {
        real[x] = bilinear((x + xmin - center + 0.5) * ss);
        total += real[x];
      }
      for (let x = 0; x < xmax; x++) {
        const w = total !== 0 ? real[x] / total : real[x];
        const scaled = w * (1 << PRECISION_BITS);
        weights[xx * ksize + x] = Math.trunc(scaled < 0 ? scaled - 0.5 : scaled + 0.5);
      }
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    return { ksize, bounds, weights };
  }

  function clip8(value) {
    const v = value >> PRECISION_BITS;
    return v < 0 ? 0 : v > 255 ? 255 : v;
  }

  // `rgba` is a flat array of inW * inH * 4 bytes, like canvas getImageData.
  // The alpha byte is ignored, as it is by Pillow's convert("RGB").
  // Returns a Uint8ClampedArray of outW * outH * 4 bytes, alpha 255.
  function resize(rgba, inW, inH, outW, outH) {
    let current = rgba;

    if (outW !== inW) {
      const { ksize, bounds, weights } = coefficients(inW, outW);
      const next = new Uint8ClampedArray(outW * inH * 4);
      for (let y = 0; y < inH; y++) {
        const row = y * inW;
        for (let xx = 0; xx < outW; xx++) {
          const xmin = bounds[xx * 2];
          const count = bounds[xx * 2 + 1];
          const k = xx * ksize;
          for (let c = 0; c < 3; c++) {
            let sum = 1 << (PRECISION_BITS - 1);  // so that the shift rounds
            for (let x = 0; x < count; x++) {
              sum += current[(row + xmin + x) * 4 + c] * weights[k + x];
            }
            next[(y * outW + xx) * 4 + c] = clip8(sum);
          }
          next[(y * outW + xx) * 4 + 3] = 255;
        }
      }
      current = next;
    }

    if (outH !== inH) {
      const { ksize, bounds, weights } = coefficients(inH, outH);
      const next = new Uint8ClampedArray(outW * outH * 4);
      for (let yy = 0; yy < outH; yy++) {
        const ymin = bounds[yy * 2];
        const count = bounds[yy * 2 + 1];
        const k = yy * ksize;
        for (let x = 0; x < outW; x++) {
          for (let c = 0; c < 3; c++) {
            let sum = 1 << (PRECISION_BITS - 1);
            for (let y = 0; y < count; y++) {
              sum += current[((ymin + y) * outW + x) * 4 + c] * weights[k + y];
            }
            next[(yy * outW + x) * 4 + c] = clip8(sum);
          }
          next[(yy * outW + x) * 4 + 3] = 255;
        }
      }
      current = next;
    }

    if (current === rgba) {  // nothing to do: still hand back a copy with alpha 255
      current = new Uint8ClampedArray(rgba);
      for (let i = 3; i < current.length; i += 4) current[i] = 255;
    }
    return current;
  }

  // torchvision Resize(size): the shorter side becomes `size`, the longer side
  // is int(size * long / short), cutting off the fraction.
  function shorterSideSize(width, height, size) {
    if (width <= height) {
      return { width: size, height: Math.floor((size * height) / width) };
    }
    return { width: Math.floor((size * width) / height), height: size };
  }

  // Python's round(): a half goes to the even number, 2.5 -> 2 and 3.5 -> 4.
  // JavaScript's Math.round(2.5) is 3, so it cannot be used for CenterCrop.
  function pythonRound(x) {
    const floor = Math.floor(x);
    const diff = x - floor;
    if (diff < 0.5) return floor;
    if (diff > 0.5) return floor + 1;
    return floor % 2 === 0 ? floor : floor + 1;
  }

  const api = { resize, shorterSideSize, pythonRound };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else globalThis.PillowResize = api;
})();

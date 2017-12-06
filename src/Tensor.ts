
type UintArray = Uint8Array | Uint16Array | Uint32Array
type IntArray = Int8Array | Int16Array | Int32Array
type FloatArray = Float32Array | Float64Array

export class Tensor {
  dims: UintArray
  strides: Uint32Array
  data: FloatArray

  constructor(
    dims: UintArray,
    data: FloatArray
  ) {
    this.data = data
    this.dims = dims
    this.strides = new Uint32Array(dims.length)

    let stride = 1

    for (var i = 0; i < dims.length; i++) {
      this.strides[i] = stride;
      stride *= dims[i];
    }

    if (data.length !== stride) {
      throw new Error(`Cannot make Tensor of ${dims.join('x')} with data of length ${data.length}`)
    }
  }

  getIndices(
    index: number
  ): Uint32Array {
    let strides = this.strides
    let len = strides.length
    let ticks = new Uint32Array(len)

    for (let i = len - 1; i >=0; i--) {
      let tick = Math.floor(index / strides[i])

      index -= tick * strides[i]

      ticks[i] = tick
    }

    return ticks
  }

  getIndex(
    indices: UintArray
  ): number {
    let strides = this.strides
    let len = strides.length

    let index = 0

    for (let i = len - 1; i >=0; i--) {
      index += indices[i] * strides[i]
    }

    return index
  }

  toNestedArrays() {
    let self = this
    let arr = []

    self.data.forEach((val, index) => {
      let temp = arr
      let indices = self.getIndices(index)

      let len = indices.length - 1

      for (let i = 0; i < len; i++) {
        let indice = indices[i]
        temp = temp[indice] = temp[indice] || []
      }

      temp[indices[len]] = val
    })

    return arr
  }

  reduce(
    func: (Float64Array) => number
  ): number {
    return func(this.data)
  }

  dimReduce (
    dim: number,
    keepdim: boolean,
    func: (Float64Array) => number
  ): Tensor {
    let dims = this.dims
    let data = this.data
    let stride = this.strides[dim]
    let size = dims[dim]

    let len = dims.length

    if (dim < 0) {
      throw new Error(`Cannot reduce negative dimension`)
    }

    if (dim >= len) {
      throw new Error(`Cannot reduce dimension ${dim} when only ${len} dimensions`)
    }

    let outSize = 1
    let outDims = keepdim ? new Uint32Array(len) : new Uint32Array(len - 1)

    for (let i = 0; i < len; i++) {
      if (i < dim) {
        outDims[i] = dims[i]
      } else if (i > dim) {
        outDims[keepdim ? i : i - 1] = dims[i]
      } else if (i == dim) {
        if (keepdim) {
          outDims[i] = 1
        }

        continue
      }

      outSize *= dims[i]
    }

    let output = new Uint32Array(outSize)

    let offset = 0

    for (let o = 0; o < outSize; o++) {

      let temp = Array(size)

      for (let i = 0; i < size; i++) {
        temp[i] = data[offset + i * stride]
      }

      output[o] = func(temp)

      offset += stride == 1 ? size : 1
    }

    return new Tensor(outDims, output)
  }
}

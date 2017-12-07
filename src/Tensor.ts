
type UintArray = Uint8Array | Uint16Array | Uint32Array
type IntArray = Int8Array | Int16Array | Int32Array
type FloatArray = Float32Array | Float64Array

type InteratorFunc = (val: number, index: number, arr: FloatArray) => void
type DimInteratorFunc = (values: FloatArray, index: number, len: number) => void
type ReducerFunc = (acc: number, val: number, index: number, arr: FloatArray) => number
type MapFunc = (val: number, index: number, arr: FloatArray) => number


class Tensor {
  static dimReduceOffset(
    index: number,
    values: number,
    stride: number
  ): number {
    return values * stride * Math.floor( index / stride ) + index % stride
  }

  static assertDim(
    dim: number,
    len: number
  ): void {
    if (dim < 0) {
      throw new Error(`Cannot reduce negative dimension`)
    } else if (dim >= len) {
      throw new Error(`Cannot reduce dimension ${dim} when only ${len} dimensions`)
    }
  }

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

  map(
    mapper: MapFunc
  ): Tensor {
    let output = this.data.map(mapper)

    return new Tensor(this.dims, output)
  }

  forEach(
    reducer: InteratorFunc
  ): void {
    this.data.forEach(reducer)
  }

  dimForEach(
    dim: number,
    iterator: DimInteratorFunc
  ): void {
    let dims = this.dims
    let data = this.data
    let stride = this.strides[dim]
    let values = dims[dim]

    let len = dims.length

    Tensor.assertDim(dim, len)

    let interations = 1

    for (let i = 0; i < len; i++) {
      if (i == dim) {
        continue
      }

      interations *= dims[i]
    }

    this._dimForEach(interations, values, stride, iterator)
  }

  _dimForEach(
    interations: number,
    values: number,
    stride: number,
    iterator: DimInteratorFunc
  ): void {
    let data = this.data

    for (let i = 0; i < interations; i++) {
      let temp = new Float64Array(values)

      let offset = Tensor.dimReduceOffset(i, values, stride)

      for (let v = 0; v < values; v++) {
        temp[v] = data[offset + v * stride]
      }

      iterator(temp, i, interations)
    }
  }

  reduce(
    reducer: ReducerFunc
  ): number {
    return this.data.reduce(reducer)
  }

  dimReduce (
    dim: number,
    keepdim: boolean,
    reducer: ReducerFunc
  ): Tensor {
    let dims = this.dims
    let data = this.data
    let stride = this.strides[dim]
    let values = dims[dim]

    let len = dims.length

    Tensor.assertDim(dim, len)

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

    let output = new Float64Array(outSize)

    this._dimForEach(outSize, values, stride, (vals, index, len) => {
      output[index] = vals.reduce(reducer)
    })

    return new Tensor(outDims, output)
  }
}

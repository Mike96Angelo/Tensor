
type UintArray = Uint8Array|Uint16Array|Uint32Array
type IntArray = Int8Array|Int16Array|Int32Array
type FloatArray = Float32Array|Float64Array

type InteratorFunc = (
  value: number,
  index: number,
  dimIndex: number,
  dimLength: number
) => void

type ReducerFunc = (
  acc: number,
  val: number
) => number

type MapFunc = (
  val: number,
  dimLength: number
) => number


class Tensor {
  static SetStrides(
    strides: UintArray,
    shape: UintArray
  ): number {
    let stride = 1

    for (let i = shape.length - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }

    return stride
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

  readonly data: FloatArray
  readonly dataShape: UintArray
  readonly dataStrides: UintArray
  readonly dataSize: number

  _shape: UintArray|null
  get shape(): UintArray {
    return this._shape || this.dataShape
  }
  _strides: UintArray
  get strides(): UintArray {
    return this._strides || this.dataStrides
  }
  _size: number|null
  get size(): number {
    return this._size || this.dataSize
  }

  constructor(
    shape: UintArray,
    data: FloatArray
  ) {
    this.data = data
    this.dataSize = data.length
    this.dataShape = shape.slice()
    this.dataStrides = new Uint32Array(shape.length)

    let size = Tensor.SetStrides(this.dataStrides, shape)

    if (data.length != size) {
      throw new Error(`Cannot make Tensor of ${shape.join('x')} with data of length ${data.length}`)
    }
  }

  getIndices(
    index: number
  ): Uint32Array {
    let strides = this.strides
    let len = strides.length
    let ticks = new Uint32Array(len)

    for (let i = 0; i < len; i++) {
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

    for (let i = len - 1; i >= 0; --i) {
      index += indices[i] * strides[i]
    }

    return index
  }

  dimOffset(
    n: number,
    dim: number
  ): number {
    let v = this.shape[dim]
    let s = this.strides[dim]

    return (v * s * Math.floor( n / s ) + n % s) % this.dataSize
  }

  dimIndex(
    i: number,
    dim: number
  ): number {
    let v = this.shape[dim]
    let s = this.strides[dim]

    i %= this.dataSize

    return s * Math.floor( i / ( v * s ) ) + i % s
  }

  _forEach(
    iterator: InteratorFunc
  ): void {
    let data = this.data
    let dataSize = this.dataSize
    let size = this.size

    for (let i = 0; i < size; i++) {
      let index = i % dataSize
      iterator(data[index], index, -1, size)
    }
  }

  forEach(
    dim: number|InteratorFunc,
    iterator?: InteratorFunc
  ): void {
    if (dim instanceof Function) {
      return this._forEach(dim)
    }

    let data = this.data
    let dataSize = this.dataSize
    let shape = this.shape
    let stride = this.strides[dim]
    let values = shape[dim]

    Tensor.assertDim(dim, shape.length)

    let interations = dataSize / values

    for (let i = 0; i < interations; i++) {
      let offset = this.dimOffset(i, dim)
      for (let v = 0; v < values; v++) {
        let index = offset + v * stride
        iterator(data[index], index, i, values)
      }
    }
  }

  _reduce(
    reducer: ReducerFunc,
    mapper: MapFunc
  ): Tensor {
    let data = this.data
    let dataSize = this.dataSize
    let size = this.size

    let outDims = new Uint32Array([1])
    let output = new Float64Array(1)

    let acc = data[0]

    for (let i = 1; i < size; i++) {
      acc = reducer(acc, data[i % dataSize])
    }

    output[0] = mapper(acc, size)

    return new Tensor(outDims, output)
  }

  reduce (
    dim: number|ReducerFunc,
    keepdim?: boolean|ReducerFunc|MapFunc,
    reducer?: ReducerFunc|MapFunc,
    mapper?: MapFunc
  ): Tensor {
    if (dim instanceof Function) {
      mapper = (keepdim instanceof Function) ? keepdim : (a) => a
      return this._reduce(dim, mapper)
    }

    if (keepdim instanceof Function) {
      mapper = reducer
      reducer = keepdim
      keepdim = false
    }

    mapper = (mapper instanceof Function) ? mapper : (a) => a

    let data = this.data
    let size = this.size
    let shape = this.shape
    let stride = this.strides[dim]
    let values = shape[dim]

    let len = shape.length

    Tensor.assertDim(dim, len)

    let outDims = keepdim ? new Uint32Array(len) : new Uint32Array(len - 1)

    for (let i = 0; i < len; i++) {
      if (i < dim) {
        outDims[i] = shape[i]
      } else if (i > dim) {
        outDims[keepdim ? i : i - 1] = shape[i]
      } else if (i == dim) {
        if (keepdim) {
          outDims[i] = 1
        }
      }
    }

    let outSize = size / values
    let output = new Float64Array(outSize)

    for (let i = 0; i < outSize; i++) {
      let offset = this.dimOffset(i, dim)

      let acc = data[offset]

      for (let v = 1; v < values; v++) {
        acc = reducer(acc, data[offset + v * stride])
      }

      output[i] = mapper(acc, values)
    }

    return new Tensor(outDims, output)
  }

  expand(...dims: number[]) {
    let len = dims.length
    let shape = new Uint32Array(dims)
    let dataShape = this.dataShape
    let shapeLen = dataShape.length

    for (let i = 1; i <= len; i++) {
      let oldSize = (dataShape[shapeLen - i] || 1)
      let newSize = shape[len - i]
      if (oldSize != 1 && newSize != oldSize) {
        throw Error(`Cannot expand where shape[n] != 1.`)
      } else if (newSize < 1) {
        throw Error(`Cannot expand to size 0.`)
      }
    }

    let strides = new Uint32Array(len)

    this._size = Tensor.SetStrides(strides, shape)
    this._shape = shape
    this._strides = strides
  }

  unexpand() {
    this._size = null
    this._shape = null
    this._strides = null
  }


  // for display purposes
  toNestedArrays() {
    let arr = []

    for (let i = 0; i < this.size; i++) {
      let temp = arr
      let indices = this.getIndices(i)

      let len = indices.length - 1

      for (let i = 0; i < len; i++) {
        let indice = indices[i]
        temp = temp[indice] = temp[indice] || []
      }

      temp[indices[len]] = this.data[i % this.dataSize]
    }

    return arr
  }
}

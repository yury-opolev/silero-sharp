using System.Buffers.Binary;
using System.Text;

namespace SileroSharp.Processing;

/// <summary>
/// Minimal reader for NumPy .npy files (float32 only).
/// </summary>
internal static class NpyReader
{
    public static float[] LoadFloat1D(string path)
    {
        using var stream = File.OpenRead(path);
        var (shape, _) = ReadHeader(stream);
        if (shape.Length != 1)
        {
            throw new InvalidOperationException($"Expected 1D array, got {shape.Length}D");
        }

        var count = shape[0];
        var bytes = new byte[count * 4];
        stream.ReadExactly(bytes);

        var result = new float[count];
        for (var i = 0; i < count; i++)
        {
            result[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        }

        return result;
    }

    public static float[,] LoadFloat2D(string path)
    {
        using var stream = File.OpenRead(path);
        var (shape, _) = ReadHeader(stream);
        if (shape.Length != 2)
        {
            throw new InvalidOperationException($"Expected 2D array, got {shape.Length}D");
        }

        var rows = shape[0];
        var cols = shape[1];
        var bytes = new byte[rows * cols * 4];
        stream.ReadExactly(bytes);

        var result = new float[rows, cols];
        var idx = 0;
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                result[r, c] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(idx));
                idx += 4;
            }
        }

        return result;
    }

    private static (int[] shape, string dtype) ReadHeader(Stream stream)
    {
        // Magic: \x93NUMPY
        Span<byte> magic = stackalloc byte[6];
        stream.ReadExactly(magic);

        // Version
        var major = stream.ReadByte();
        var minor = stream.ReadByte();

        // Header length
        int headerLen;
        if (major >= 2)
        {
            Span<byte> lenBytes = stackalloc byte[4];
            stream.ReadExactly(lenBytes);
            headerLen = BinaryPrimitives.ReadInt32LittleEndian(lenBytes);
        }
        else
        {
            Span<byte> lenBytes = stackalloc byte[2];
            stream.ReadExactly(lenBytes);
            headerLen = BinaryPrimitives.ReadInt16LittleEndian(lenBytes);
        }

        // Header string (Python dict)
        var headerBytes = new byte[headerLen];
        stream.ReadExactly(headerBytes);
        var header = Encoding.ASCII.GetString(headerBytes).Trim();

        // Parse shape from header like: {'descr': '<f4', 'fortran_order': False, 'shape': (126523, 16), }
        var shapeStart = header.IndexOf("'shape': (", StringComparison.Ordinal) + "'shape': (".Length;
        var shapeEnd = header.IndexOf(')', shapeStart);
        var shapeStr = header[shapeStart..shapeEnd].Trim().TrimEnd(',');

        int[] shape;
        if (string.IsNullOrEmpty(shapeStr))
        {
            shape = [];
        }
        else
        {
            shape = shapeStr.Split(',', StringSplitOptions.TrimEntries)
                .Select(int.Parse)
                .ToArray();
        }

        // Parse dtype
        var descrStart = header.IndexOf("'descr': '", StringComparison.Ordinal) + "'descr': '".Length;
        var descrEnd = header.IndexOf('\'', descrStart);
        var dtype = header[descrStart..descrEnd];

        return (shape, dtype);
    }
}

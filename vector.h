#ifndef VECTOR_H

#include <stdint.h>
#include <math.h>

// SECTION START: vector2
struct vector2
{
	float X;
	float Y;
};

inline vector2 Vector2(float X, float Y)
{
	vector2 Result;
	Result.X = X;
	Result.Y = Y;
	return Result;
}

inline vector2 Vector2(int32_t X, int32_t Y)
{
	vector2 Result;
	Result.X = (float) X;
	Result.Y = (float) Y;
	return Result;
}

inline vector2 Vector2(uint32_t X, uint32_t Y)
{
	vector2 Result;
	Result.X = (float) X;
	Result.Y = (float) Y;
	return Result;
}

inline vector2 operator*(float A, vector2 B)
{
	vector2 Result;
	Result.X = A * B.X;
	Result.Y = A * B.Y;
	return Result;
}

inline vector2 operator*(vector2 B, float A)
{
	vector2 Result;
	Result.X = A * B.X;
	Result.Y = A * B.Y;
	return Result;	
}

inline vector2 operator*=(vector2 &B, float A)
{
	B = A * B;
	return B;
}

inline vector2 operator/(vector2 B, float A)
{
	return (1 / A) * B;
}

inline vector2 operator-(vector2 A)
{
	vector2 Result;
	Result.X = -1 * A.X;
	Result.Y = -1 * A.Y;
	return Result;
}

inline vector2 operator+(vector2 A, vector2 B)
{
	vector2 Result;
	Result.X = A.X + B.X;
	Result.Y = A.Y + B.Y;
	return Result;
}

inline vector2 operator+=(vector2 &A, vector2 B)
{
	A = A + B;
	return A; 
}

inline vector2 operator-(vector2 A, vector2 B)
{
	vector2 Result;
	Result.X = A.X - B.X;
	Result.Y = A.Y - B.Y;
	return Result;
}

inline vector2 operator-=(vector2 &A, vector2 B)
{
	A = A - B;
	return A;
}

inline vector2 Abs(vector2 A)
{
	vector2 Result;
	Result.X = (float) fabs(A.X);
	Result.Y = (float) fabs(A.Y);
	return Result;
}

inline float Inner(vector2 A, vector2 B)
{
	return (A.X * B.X) + (A.Y * B.Y);
}

inline float MagnitudeSquared(vector2 A)
{
	return Inner(A, A);
}

inline float Magnitude(vector2 A)
{
	return sqrtf(MagnitudeSquared(A));
}

inline vector2 Normalize(vector2 A)
{
	return (1.0f / Magnitude(A)) * A;	
}

inline vector2 Perpendicular(vector2 A)
{
	vector2 Result;
	Result.X = -A.Y;
	Result.Y = A.X;
	return Result;	
}

inline vector2 Hadamard(vector2 A, vector2 B)
{
	vector2 Result;
	Result.X = A.X * B.X;
	Result.Y = A.Y * B.Y;
	return Result;
}
// SECTION STOP: vector2

// SECTION START: vector3
union vector3
{
	struct
	{
		float X;
		float Y;
		float Z;
	};
	struct
	{
		vector2 Xy;
		float Z;
	};
};

inline vector3 Vector3(float X, float Y, float Z)
{
	vector3 Result;
	Result.X = X;
	Result.Y = Y;
	Result.Z = Z;
	return Result;
}

inline vector3 Vector3(vector3 V, float Z)
{
	vector3 Result;
	Result.X = V.X;
	Result.Y = V.Y;
	Result.Z = Z;
	return Result;
}

inline vector3 operator*(float A, vector3 B)
{
	vector3 Result;
	Result.X = A * B.X;
	Result.Y = A * B.Y;
	Result.Z = A * B.Z;
	return Result;
}

inline vector3 operator*(vector3 B, float A)
{
	vector3 Result;
	Result.X = A * B.X;
	Result.Y = A * B.Y;
	Result.Z = A * B.Z;
	return Result;
}

inline vector3 operator*=(vector3 &B, float A)
{
	B = A * B;
	return B;
}

inline vector3 operator/(vector3 B, float A)
{
	return (1 / A) * B;
}

inline vector3 operator-(vector3 A)
{
	vector3 Result;
	Result.X = -1 * A.X;
	Result.Y = -1 * A.Y;
	Result.Z = -1 * A.Z;
	return Result;
}

inline vector3 operator+(vector3 A, vector3 B)
{
	vector3 Result;
	Result.X = A.X + B.X;
	Result.Y = A.Y + B.Y;
	Result.Z = A.Z + B.Z;
	return Result;
}

inline vector3 operator+=(vector3 &A, vector3 B)
{
	A = A + B;
	return A; 
}

inline vector3 operator-(vector3 A, vector3 B)
{
	vector3 Result;
	Result.X = A.X - B.X;
	Result.Y = A.Y - B.Y;
	Result.Z = A.Z - B.Z;
	return Result;
}

inline vector3 operator-=(vector3 &A, vector3 B)
{
	A = A - B;
	return A;
}

inline float Inner(vector3 A, vector3 B)
{
	return (A.X * B.X) + (A.Y * B.Y) + (A.Z * B.Z);
}

inline float MagnitudeSquared(vector3 A)
{
	return Inner(A, A);
}

inline float Magnitude(vector3 A)
{
	return sqrtf(MagnitudeSquared(A));
}

inline vector3 Normalize(vector3 A)
{
	return (1.0f / Magnitude(A)) * A; 
}

inline vector3 Lerp(vector3 A, float T, vector3 B)
{
    vector3 Result = (1.0f - T) * A + T * B;

    return Result;
}

inline vector3 Hadamard(vector3 A, vector3 B)
{
	vector3 Result;
	Result.X = A.X * B.X;
	Result.Y = A.Y * B.Y;
	Result.Z = A.Z * B.Z;
	return Result;
}
// SECTION STOP: vector3

// SECTION START: vector4
union vector4
{
	struct
	{
		float X;
		float Y;
		float Z;
		float W;
	};
	struct
	{
		float R;
		float G;
		float B;
		float A;
	};
	struct
	{
		vector3 Rgb;
		float A;
	};
	struct
	{
		vector3 Xyz;
		float W;
	};
	struct
	{
		vector2 Xy;
		float Z;
		float W;
	};
	struct
	{
		float X;
		vector2 Yz;
		float W;
	};
	struct
	{
		float X;
		float Y;
		vector2 Zw;
	};
};

inline vector4 Vector4(float X, float Y, float Z, float W)
{
	vector4 Result;
	Result.X = X;
	Result.Y = Y;
	Result.Z = Z;
	Result.W = W;
	return Result;
}

inline vector4 ToVector4(vector3 Vector3, float W)
{
	vector4 Result;
	Result.X = Vector3.X;
	Result.Y = Vector3.Y;
	Result.Z = Vector3.Z;
	Result.W = W;
	return Result;
}

inline vector4 operator*(float A, vector4 B)
{
	vector4 Result;
	Result.X = A * B.X;
	Result.Y = A * B.Y;
	Result.Z = A * B.Z;
	Result.W = A * B.W;
	return Result;
}

inline vector4 operator*(vector4 B, float A)
{
	vector4 Result;
	Result.X = A * B.X;
	Result.Y = A * B.Y;
	Result.Z = A * B.Z;
	Result.W = A * B.W;
	return Result;
}

inline vector4 operator*=(vector4 &B, float A)
{
	B = A * B;
	return B;
}

inline vector4 operator-(vector4 A)
{
	vector4 Result;
	Result.X = -1 * A.X;
	Result.Y = -1 * A.Y;
	Result.Z = -1 * A.Z;
	Result.W = -1 * A.W;
	return Result;
}

inline vector4 operator+(vector4 A, vector4 B)
{
	vector4 Result;
	Result.X = A.X + B.X;
	Result.Y = A.Y + B.Y;
	Result.Z = A.Z + B.Z;
	Result.W = A.W + B.W;
	return Result;
}

inline vector4 operator+=(vector4 &A, vector4 B)
{
	A = A + B;
	return A; 
}

inline vector4 operator-(vector4 A, vector4 B)
{
	vector4 Result;
	Result.X = A.X - B.X;
	Result.Y = A.Y - B.Y;
	Result.Z = A.Z - B.Z;
	Result.W = A.W - B.W;
	return Result;
}

inline vector4 operator-=(vector4 &A, vector4 B)
{
	A = A - B;
	return A;
}

inline vector4 Lerp(vector4 A, float T, vector4 B)
{
    vector4 Result = (1.0f - T) * A + T * B;

    return Result;
}

inline vector4 Hadamard(vector4 A, vector4 B)
{
	vector4 Result;
	Result.X = A.X * B.X;
	Result.Y = A.Y * B.Y;
	Result.Z = A.Z * B.Z;
	Result.W = A.W * B.W;
	return Result;
}
// SECTION STOP vector4

#define VECTOR_H
#endif
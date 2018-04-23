//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2018-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Papp Kristof
// Neptun : ZP3ZTV
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#define _USE_MATH_DEFINES		// M_PI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;
const unsigned int MAXDEPTH = 5;
bool START_OUT = false;
const unsigned int screenWidth = windowWidth/2, screenHeight = windowHeight/2;

struct vec3 {
	float x, y, z;
	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }
	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }
	vec3 operator/(float d) const { return vec3(x / d, y / d, z / d); }
	vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	void operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; }
	vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator/(const vec3& v) const { return vec3(x / v.x, y / v.y, z / v.z); }
	vec3 operator-() const { return vec3(-x, -y, -z); }
	vec3 normalize() const { return (*this) * (1 / (Length() + 0.000001)); }
	float Length() const { return sqrtf(x * x + y * y + z * z); }
	operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

class Material {
public:
	vec3 color; //Szín (ka, kd)
	vec3 kappa; //Kioltási tényezõ (ks)
	vec3 n; //Törésmutató
	float  shininess; //Fényesség, csillogás (fordítottan arányos)
	bool isTextured = false; //Textúrázott-e

	Material(vec3 col, vec3 k, float _shininess) : color(col), kappa(k) 
	{ 
		shininess = _shininess; 
	}

	bool virtual isReflective() { return false; } //Fényvisszaverõ
	bool virtual isRefractive() { return false; } //Fénytörõ
	bool virtual isRough() { return true; } //Rücskös anyag

	//Textúrázott szín
	vec3 Color(vec3 pos)
	{
		//Ha nem textúrázott az anyag:
		if (!isTextured) return color; 

		//Ha textúrázott:
		float distance = pos.Length(); //Az origótól vett távolság
		if (fmod(distance, 3.0f) > (3.0f / 2.0f)) {
			return color;
		}
		else {
			return color*0.35; //Sötétebb
		}
	}

	//Árnyékoló fv
	vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad, vec3 pos) {
		vec3 reflRad(0, 0, 0);
		float cosTheta = dot(normal, lightDir);
		if (cosTheta < 0) return reflRad;
		reflRad = inRad * Color(pos) * cosTheta; //Textúrához figyelembe veszi a helyet
		
		vec3 halfway = (viewDir + lightDir).normalize();
		float cosDelta = dot(normal, halfway);
		if (cosDelta < 0) return reflRad;
		return reflRad + inRad * kappa * pow(cosDelta, shininess);
	}
};

class Gold : public Material {

public:
	Gold(vec3 col, vec3 k, float _shininess) : Material(col, k, _shininess) 
	{
		n = vec3(0.17f, 0.35f, 1.5f);
		kappa = vec3(3.1f, 2.7f, 1.9f);
	}

	bool isReflective() { return true; } //Fényvisszaverõ
	bool isRefractive() { return false; } //Fénytörõ
	bool isRough() { return false; } //Rücskös anyag
};

class Glass : public Material {

public:
	Glass(vec3 col, vec3 k, float _shininess) : Material(col, k, _shininess) 
	{
		color = vec3(0, 0, 0);
		kappa = vec3(0, 0, 0);
		n = vec3(1.5f, 1.5f, 1.5f);
	}

	bool isReflective() { return true; } //Fényvisszaverõ
	bool isRefractive() { return true; } //Fénytörõ
	bool isRough() { return false; } //Rücskös anyag
};

class Water : public Material {

public:
	Water(vec3 col, vec3 k, float _shininess) : Material(col, k, _shininess)
	{
		color = vec3(0, 0, 0);
		kappa = vec3(0, 0, 0);
		n = vec3(1.3f, 1.3f, 1.3f);
	}

	bool isReflective() { return true; } //Fényvisszaverõ
	bool isRefractive() { return true; } //Fénytörõ
	bool isRough() { return false; } //Rücskös anyag
};

struct Hit {
	float t; //A szemtõl mért távolság ha negatív, akkor nem látjuk
	vec3 position; //Metszéspont
	vec3 normal; //A felület normálvektora
	Material * material; //A felület anyaga
	Hit() { t = -1; } //Alapból nincs találat
};

struct Ray {
	vec3 start, dir;
	bool out; //Kívül van-e?
	Ray(vec3 _start, vec3 _dir, bool _out=START_OUT) { start = _start; dir = _dir.normalize(); out = _out; }
};


class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Flat : public Intersectable {
public:
	vec3 point;
	vec3 normal;

	Flat(vec3 p, vec3 n, Material* _material) {
		point = p;
		normal = n;
		material = _material;
		material->isTextured = true;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		float tmp = dot(normal,ray.dir); 

		if (tmp == 0) {return hit;} //ha merõlegesek, akkor nincs metszéspont

		hit.material = material;
		hit.normal = normal;

		float t = dot((point - ray.start),normal) / tmp;
		if (t < 0) {
			return hit;
		}
		hit.position = ray.dir * t + ray.start; //A megtett út
		hit.t = t;
		return hit;
	}
};

struct Sphere : public Intersectable {
public:
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}
	Hit intersect(const Ray& ray) {
		Hit hit; //Találat helye, anyaga
		vec3 dist = ray.start - center; //A szem és gömb közepének távolsága
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0;
		float c = dot(dist, dist) - radius * radius;

		//Másodfokú egyenlet gyökei (t1, t2)
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit; //Nincs találat
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;

		//Kisebb pozitív gyök kiválasztása
		if (t1 <= 0 && t2 <= 0) return hit; //Nincs találat
		if (t1 <= 0 && t2 > 0)       hit.t = t2;
		else if (t2 <= 0 && t1 > 0)  hit.t = t1;
		else if (t1 < t2)            hit.t = t1;
		else                         hit.t = t2;

		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		if (material->isRefractive() && ray.out) hit.normal = hit.normal*(-1);
		hit.material = material;
		return hit;
	}
};

struct Paraboloid : public Intersectable {
	//Egy síktól és egy ponttól azonos távolságra lévõ pontok halmaza
	vec3 focus; //A fókuszpont
	vec3 point; //A sík egy pontja
	vec3 normal; //A sík normálvektora

	Paraboloid(vec3 _focus, vec3 _point, vec3 _normal, Material* _material) {
		focus = _focus;
		point = _point;
		normal = _normal;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit; //A találat

		vec3 distFocus = ray.start - focus; //A fókusz távolsága a szemtõl
		vec3 distPoint = ray.start - point; //A pont távolsága a szemtõl

		//Másodfokú egyenlet paraméterei (a, b, c)
		float a = dot(ray.dir, ray.dir) - dot(normal, ray.dir)*dot(normal, ray.dir);
		float b = (dot(distFocus, ray.dir) - dot(normal, ray.dir)*dot(normal, distPoint))*2.0f;
		float c = dot(distFocus, distFocus) - dot(normal, distPoint)*dot(normal, distPoint);

		//Másodfokú egyenlet gyökei (t1, t2)
		float discr = b * b - 4.0 * a * c; 
		if (discr < 0) return hit; //Nincs megoldás
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;
		float t2 = (-b - sqrt_discr) / 2.0 / a;

		//Kisebb pozitív gyök kiválasztása
		if (t1 <= 0 && t2 <= 0) return hit; //Nincs találat
		if (t1 <= 0 && t2 > 0)       hit.t = t2;
		else if (t2 <= 0 && t1 > 0)  hit.t = t1;
		else if (t1 < t2)            hit.t = t1;
		else                         hit.t = t2;

		hit.position = ray.start + ray.dir * hit.t; //A találat helye
		hit.material = material; //A találat anyaga megegyezik a paraboloid anyagával
		hit.normal = vec3((hit.position.x - focus.x) - (normal.x*(hit.position.x - point.x)), (hit.position.y - focus.y) - (normal.y*(hit.position.y - point.y)), (hit.position.z - focus.z) - (normal.z*(hit.position.z - point.z))).normalize();
		if (material->isRefractive() && ray.out) hit.normal = hit.normal*(-1);

		return hit;
	}
};

class Camera{
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = w.Length();
		right = cross(vup, w).normalize() * f * tan(fov / 2);
		up = cross(w, right).normalize() * f * tan(fov / 2);
	}
	Ray getray(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / screenWidth - 1) + up * (2.0 * (Y + 0.5) / screenHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

class FishCamera {
	vec3 pos;

public:
	void set(vec3 _pos, vec3 a, vec3 b, vec3 c) {
		pos = _pos;
	}

	Ray getray(int px, int py) {

		float beta = ((float)py / screenHeight) * M_PI;
		float alpha = ((screenWidth - px) / (float)screenWidth) * 2.0f * M_PI;

		//Polárkoordináták
		//https://hu.wikipedia.org/wiki/G%C3%B6mbi_koordin%C3%A1t%C3%A1k

		float x = sinf(beta) * cosf(alpha);
		float y = sinf(beta) * sinf(alpha);
		float z = -cosf(beta);
		vec3 dir(x, y, z);

		//A halszembõl kiinduló sugár
		return Ray(pos, dir.normalize());
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = _direction.normalize();
		Le = _Le;
	}
};

struct PointLight {
	vec3 position; //Hely
	vec3 Le; //Intenzitás (a 0 távolságban)

	PointLight(vec3 _pos, vec3 _Le) {
		position = _pos;
		Le = _Le;
	}

	vec3 Intenz(vec3 place) {
		vec3 one(1, 1, 1);
		//Az intenzitás a távolság négyzetével fordítottan arányosan csökken
		return (one /((position - place).Length()*(position - place).Length()))*Le;
	}
};

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<PointLight *> lights;
	Camera camera;
	FishCamera fishCamera;
	vec3 La;
public:
	void build(bool parab) {
		vec3 eye = vec3(0, -17, 1.5f);
		vec3 vup = vec3(0, 0, 1);
		vec3 lookat = vec3(0, 5, 5.5f);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		fishCamera.set(vec3(0, -14, 1.5f), lookat, vup, fov);
		La = vec3(0.1f, 0.1f, 0.1f); //Ambiens fény
		lights.push_back(new PointLight(vec3(1, -5, 2), vec3(5, 5, 5)));
		lights.push_back(new PointLight(vec3(2, -8, 5), vec3(15, 15, 15)));
		lights.push_back(new PointLight(vec3(2, -28, 5), vec3(6, 7, 6)));

		vec3 color0(0.25f, 0.15f, 0.0f);
		vec3 ks(5, 5, 5); //Kappa

		vec3 color1(0.6f, 0.1f, 0.1f);
		vec3 color2(0.1f, 0.7f, 0.1f);
		vec3 color3(0.1f, 0.1f, 0.8f);
		vec3 color4(0.1f, 0.5f, 0.4f);
		vec3 color5(0.6f, 0.6f, 0.0f);
		vec3 color6(0.6f, 0.1f, 0.7f);
	
		//Gömbök aranyból
		objects.push_back(new Sphere(vec3(-7, 3, 0), 2.0f, new Gold(color0, ks, 50)));
		objects.push_back(new Sphere(vec3(7, 3, 0), 2.0f, new Gold(color0, ks, 50)));
		objects.push_back(new Sphere(vec3(-7, 3, 10), 2.0f, new Gold(color0, ks, 50)));
		objects.push_back(new Sphere(vec3(7, 3, 10), 2.0f, new Gold(color0, ks, 50)));

		//Paraboloidok aranyból (lefelé és felfelé nyitott)
		objects.push_back(new Paraboloid(vec3(1, -1, 4), vec3(1, -1, 5), vec3(0, 0, 1), new Gold(color0, ks, 50)));
		objects.push_back(new Paraboloid(vec3(1, -1, 5), vec3(1, -1, 4), vec3(0, 0, 1), new Gold(color0, ks, 50)));

		//Paraboloid akvárium (vízbõl)
		if (parab == true)
		{
			objects.push_back(new Paraboloid(vec3(0, -14, 0), vec3(0, -14, -1), vec3(0, 0, 1), new Water(color0, ks, 500)));
		}

		//Plafon piros
		objects.push_back(new Flat(vec3(10, 10, 10), vec3(0, 0, -1), new Material(color1, ks, 1000)));

		//Bal fal sötétzöld
		objects.push_back(new Flat(vec3(10, 10, 10), vec3(-1, 0, 0), new Material(color2, ks, 1000)));

		//Padló sötétkék
		objects.push_back(new Flat(vec3(0, 0, 0), vec3(0, 0, 1), new Material(color3, ks, 1000)));

		//Jobb fal világoszöld
		objects.push_back(new Flat(vec3(-10, 10, 10), vec3(1, 0, 0), new Material(color4, ks, 1000)));

		//Elsõ fal sárga
		objects.push_back(new Flat(vec3(-10, 10, 10), vec3(0, -1, 0), new Material(color5, ks, 1000)));

		//Hátsó fal lila
		objects.push_back(new Flat(vec3(-10, -30, 10), vec3(0, 1, 0), new Material(color6, ks, 1000)));
	}

	void render(vec3 image[]) {
#pragma omp parallel for
		for (int Y = 0; Y < screenHeight; Y++) {
			for (int X = 0; X < screenWidth; X++) {
				if (X < screenWidth / 2) { //Elsõ negyed
					image[Y * screenWidth + X] = trace(camera.getray(2*X, Y));
				}
				if (X > screenWidth / 2) { //Második negyed
					image[Y * screenWidth + X] = trace(fishCamera.getray(2*X - screenWidth, Y));
				}
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		return bestHit;
	}

	//Közvetlen megvilágítás
	vec3 DirectLight(Hit hit, Ray ray) {
		vec3 outRadiance = hit.material->Color(hit.position) * La;
			for (PointLight * light : lights) {
				Ray shadowRay(hit.position + hit.normal*0.0001f, light->position - hit.position);
				Hit shadowHit = firstIntersect(shadowRay);
				if (shadowHit.t < 0 || shadowHit.t > (hit.position - light->position).Length())
					outRadiance += hit.material->shade(hit.normal, -ray.dir, light->position-hit.position, light->Intenz(hit.position), hit.position);
			}
			return outRadiance;
	}

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(normal, inDir) * 2.0f;
	};

	vec3 refract(vec3 inDir, vec3 normal, float ns) {
		float cosa = -dot(inDir, normal);
		float disc = 1 - (1 - cosa * cosa) / ns / ns; // scalar n
		if (disc <= 0) return vec3(0, 0, 0);
		return (inDir / ns) + normal * (cosa / ns - sqrt(disc));
	};

	vec3 Fresnel(vec3 inDir, vec3 normal, Material* m) {
		vec3 kappa = m->kappa; //Kioltási tényezõ
		vec3 n = m->n; //Törésmutató
	
		float cosa = -dot(inDir, normal);
		vec3 one(1, 1, 1);

		vec3 F0 = ((n - one)*(n - one) + kappa*kappa)/ ((n + one)*(n + one) + kappa * kappa);
		vec3 ret = (one-F0)*pow(1-cosa,5);
		ret = ret + F0;
		return ret;
	}

	vec3 trace(Ray ray, int d = 0) {
		if (d > MAXDEPTH) return La;
		Hit hit = firstIntersect(ray); //Az elõször metszett objektum
		if (hit.t < 0) return La; //Ha nem látunk semmit
		vec3 outRad(0,0,0);

		//if (hit.material->isRough()) 
			outRad = DirectLight(hit, ray);

		if (hit.material->isReflective()) {
			vec3 reflectionDir = reflect(ray.dir, hit.normal);
			Ray reflectRay(hit.position + hit.normal*0.0001f, reflectionDir, ray.out);
			outRad += trace(reflectRay, d + 1)*Fresnel(ray.dir, hit.normal, hit.material);
		}

		if (hit.material->isRefractive()) {
			float ior = (ray.out) ? hit.material->n.x : 1 / hit.material->n.x;
			vec3 normal = hit.normal;
			vec3 refractionDir = refract(ray.dir, normal, ior);
			if (refractionDir.Length() > 0) {
				Ray refractRay(hit.position - normal*0.0001f, refractionDir, !ray.out);
				vec3 tmp = vec3(1, 1, 1) - Fresnel(ray.dir, hit.normal, hit.material);
				tmp = tmp * trace(refractRay, d + 1);
				outRad += tmp;
			}
		}
		return outRad;
	}
};

Scene topScene;
Scene bottomScene;

void getErrorInfo(unsigned int handle) {
	int logLen, written;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
	unsigned int vao, textureId;	// vertex array object id and texture id
public:
	void Create(vec3 image[]) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

																	  // Create objects by setting up their vertex data on the GPU
		glGenTextures(1, &textureId);  				// id generation
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenWidth, screenHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(shaderProgram, "textureUnit");
		if (location >= 0) {
			glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;
vec3 topImage[screenWidth * screenHeight * 2];	//A felsõ 2 kép
vec3 bottomImage[screenWidth * screenHeight * 2]; //Az alsó két kép
vec3 image[windowWidth * windowHeight]; // The image, which stores the ray tracing result

void CombineIntoOne(vec3* bottom, vec3* top) {
	for (int Y = 0; Y < windowHeight; Y++) {
		for (int X = 0; X < windowWidth; X++) {
			if (Y > screenHeight) {
				image[Y * screenWidth + X - screenHeight * screenWidth / 2] = top[(Y - screenHeight) * windowWidth + X]; //Top
			}
			else {
				image[Y * screenWidth + X] = bottom[Y * windowWidth + X]; //Bottom
			}
		}
	}
}

// Initialization, create an OpenGL context
void onInitialization() {
		glViewport(0, 0, windowWidth, windowHeight);
		topScene.build(false);
		topScene.render(topImage);

		bottomScene.build(true);
		bottomScene.render(bottomImage);

		CombineIntoOne(bottomImage, topImage);

		fullScreenTexturedQuad.Create(image);

		// Create vertex shader from string
		unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
		if (!vertexShader) { printf("Error in vertex shader creation\n"); exit(1); }
		glShaderSource(vertexShader, 1, &vertexSource, NULL);
		glCompileShader(vertexShader);
		checkShader(vertexShader, "Vertex shader error");

		// Create fragment shader from string
		unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fragmentShader) { printf("Error in fragment shader creation\n"); exit(1); }
		glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
		glCompileShader(fragmentShader);
		checkShader(fragmentShader, "Fragment shader error");

		// Attach shaders to a single program
		shaderProgram = glCreateProgram();
		if (!shaderProgram) { printf("Error in shader program creation\n"); exit(1); }
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);

		// Connect the fragmentColor to the frame buffer memory
		glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																	// program packaging
		glLinkProgram(shaderProgram);
		checkLinking(shaderProgram);
		glUseProgram(shaderProgram); 	// make this program run
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
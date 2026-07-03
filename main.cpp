#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <random>
#include <ctime>
#include <fstream>  // для записи в файл

// Перечисление для типов шума
enum NoiseType {
    DEPOLARIZING,      // Деполяризующий шум
    AMPLITUDE_DAMPING, // Амплитудное затухание
    PHASE_DAMPING,     // Фазовый шум (дефазировка)
    RANDOM_NOISE,      // Случайный шум
    COHERENT_NOISE     // Когерентный шум (ошибки вращения)
};

// Версия матрицы на указателях
// элементы хранятся в одном массиве data_ в row-major:
// data_[r * cols_ + c].
class MatrixPtr {
public:
    using Complex = std::complex<double>;

    // Конструктор должен выделяет память под rows * cols элементов и заполняет init.
    MatrixPtr(std::size_t rows, std::size_t cols, Complex init = Complex{0.0, 0.0})
            : rows_(rows), cols_(cols), data_(new Complex[rows * cols]) {
        for (std::size_t i = 0; i < rows_ * cols_; ++i) {
            data_[i] = init;
        }
    }

    // копирование в свою память.
    MatrixPtr(const MatrixPtr& other)
            : rows_(other.rows_), cols_(other.cols_), data_(new Complex[other.rows_ * other.cols_]) {
        for (std::size_t i = 0; i < rows_ * cols_; ++i) {
            data_[i] = other.data_[i];
        }
    }

    // Перемещение забираем указатель у other.
    MatrixPtr(MatrixPtr&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
    }

    // Copy-and-swap присваивание
    MatrixPtr& operator=(MatrixPtr other) {
        swap(other);
        return *this;
    }

    // Освобождение памяти.
    ~MatrixPtr() { delete[] data_; }

    // Обмен внутренним состоянием.
    void swap(MatrixPtr& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(data_, other.data_);
    }

    // Размеры матрицы.
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // Доступ к элементу с проверкой границ
    Complex& operator()(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data_[r * cols_ + c];
    }

    // константный доступ к элементу
    const Complex& operator()(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data_[r * cols_ + c];
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    Complex* data_;
};

// матричное умножение
// C = A * B, C(i, j) = sum_k A(i, k) * B(k, j).
MatrixPtr multiply(const MatrixPtr& a, const MatrixPtr& b) {
    // Для (m x n) * (n x p) внутренние размерности должны совпасть
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("multiply: incompatible matrix sizes");
    }

    MatrixPtr result(a.rows(), b.cols(), MatrixPtr::Complex{0.0, 0.0});
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const MatrixPtr::Complex aik = a(i, k);
            for (std::size_t j = 0; j < b.cols(); ++j) {
                result(i, j) += aik * b(k, j);
            }
        }
    }
    return result;
}

// Тензорное произведение A  B.
// Если A (m x n), B (p x q), то результат (m*p x n*q):
// result[i*p + k, j*q + l] = A[i, j] * B[k, l].
MatrixPtr kroneckerProduct(const MatrixPtr& a, const MatrixPtr& b) {
    MatrixPtr result(a.rows() * b.rows(), a.cols() * b.cols(), MatrixPtr::Complex{0.0, 0.0});

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            for (std::size_t k = 0; k < b.rows(); ++k) {
                for (std::size_t l = 0; l < b.cols(); ++l) {
                    result(i * b.rows() + k, j * b.cols() + l) = a(i, j) * b(k, l);
                }
            }
        }
    }

    return result;
}

// y = A x, где A (m x n), x (n), y (m).
// Возвращается новый динамический массив; вызывающий код обязан сделать delete[]
static MatrixPtr::Complex* multiplyByVector(const MatrixPtr& a,
                                            const MatrixPtr::Complex* x,
                                            std::size_t xSize) {
    if (a.cols() != xSize) {
        throw std::invalid_argument("multiplyByVector: incompatible sizes");
    }

    auto* y = new MatrixPtr::Complex[a.rows()];
    for (std::size_t i = 0; i < a.rows(); ++i) {
        y[i] = MatrixPtr::Complex{0.0, 0.0};
        for (std::size_t j = 0; j < a.cols(); ++j) {
            y[i] += a(i, j) * x[j];
        }
    }
    return y;
}

// y = A^H x, где A^H - эрмитово-сопряженная матрица.
// Размерности: A (m x n), x (m), y (n).
// Возвращается новый динамический массив;
static MatrixPtr::Complex* multiplyByConjugateTransposeVector(const MatrixPtr& a,
                                                              const MatrixPtr::Complex* x,
                                                              std::size_t xSize) {
    if (a.rows() != xSize) {
        throw std::invalid_argument("multiplyByConjugateTransposeVector: incompatible sizes");
    }

    auto* y = new MatrixPtr::Complex[a.cols()];
    for (std::size_t i = 0; i < a.cols(); ++i) {
        y[i] = MatrixPtr::Complex{0.0, 0.0};
        for (std::size_t j = 0; j < a.rows(); ++j) {
            y[i] += std::conj(a(j, i)) * x[j];
        }
    }
    return y;
}

// Евклидова норма комплексного вектора:
// ||x||_2 = sqrt(sum_i |x_i|^2).
static double vectorNorm2(const MatrixPtr::Complex* x, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += std::norm(x[i]);
    }
    return std::sqrt(sum);
}

// Спектральная норма матрицы:
// ||A||_2 = sigma_max(A) = sqrt(lambda_max(A^H A)).
//
// Численно используем степенной метод для M = A^H A:
// 1) y = A x
// 2) z = A^H y = A^H A x
// 3) x <- z / ||z||
// При сходимости ||z|| -> lambda_max(A^H A), поэтому результат sqrt(lambda)
double spectralNorm(const MatrixPtr& a, std::size_t maxIters = 1000, double tol = 1e-12) {
    // Для пустой матрицы норма 0.
    if (a.cols() == 0 || a.rows() == 0) {
        return 0.0;
    }

    // Начальный вектор в пространстве столбцов A
    const std::size_t n = a.cols();
    auto* x = new MatrixPtr::Complex[n];
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = MatrixPtr::Complex{1.0, 0.0};
    }

    // Нормализуем x, чтобы избежать переполнений/затухания на старте
    const double initNorm = vectorNorm2(x, n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] /= initNorm;
    }

    // Предыдущее приближение lambda_max(A^H A)
    double lambdaPrev = 0.0;
    for (std::size_t iter = 0; iter < maxIters; ++iter) {
        MatrixPtr::Complex* y = multiplyByVector(a, x, n);                          // y = A x
        MatrixPtr::Complex* z = multiplyByConjugateTransposeVector(a, y, a.rows()); // z = A^H A x
        delete[] y;

        const double zNorm = vectorNorm2(z, n);
        // Если z занулен, значит в текущем направлении нет компоненты дающей рост
        // Для этой задачи это эквивалентно нулевой оценке нормы
        if (zNorm == 0.0) {
            delete[] z;
            delete[] x;
            return 0.0;
        }

        // Нормируем новый вектор степени
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = z[i] / zNorm;
        }
        delete[] z;

        // Текущая оценка максимального собственного значения A^H A
        const double lambda = zNorm;
        // Критерий остановки по относительному изменению
        if (std::abs(lambda - lambdaPrev) < tol * std::max(1.0, lambda)) {
            delete[] x;
            return std::sqrt(lambda);
        }
        lambdaPrev = lambda;
    }

    delete[] x;
    return std::sqrt(lambdaPrev);
}

// вывод матриц
void printMatrix(const MatrixPtr& m, const char* title) {
    std::cout << title << '\n';
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            std::cout << std::setw(12) << m(i, j) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

MatrixPtr Psi(const MatrixPtr& a, const MatrixPtr& b) {
    // Проверяем, что это векторы-столбцы
    if (a.cols() != 1 || b.cols() != 1) {
        throw std::invalid_argument("Psi: inputs must be column vectors");
    }

    MatrixPtr result(a.rows() * b.rows(), 1, MatrixPtr::Complex{0.0, 0.0});

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t k = 0; k < b.rows(); ++k) {
            result(i * b.rows() + k, 0) = a(i, 0) * b(k, 0);
        }
    }

    // Нормализация
    double dimension = static_cast<double>(a.rows() * b.rows());
    double normalizationFactor = 1.0 / std::sqrt(dimension);
    for (std::size_t i = 0; i < result.rows(); ++i) {
        result(i, 0) = result(i, 0) * normalizationFactor;
    }

    return result;
}

// произведение кет фи на бра фи из столбца фи
MatrixPtr outerProduct(const MatrixPtr& phi) {
    std::size_t N = phi.rows();
    if (phi.cols() != 1) {
        throw std::invalid_argument("outerProduct: input must be a column vector (N x 1)");
    }

    MatrixPtr result(N, N, MatrixPtr::Complex{0.0, 0.0});
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            result(i, j) = phi(i, 0) * std::conj(phi(j, 0));
        }
    }
    return result;
}

// Главная функция: ро = (I на E тензорно) · кет фи на бра фи тензорно
MatrixPtr RhoKsi(const MatrixPtr& I, const MatrixPtr& E, const MatrixPtr& a, const MatrixPtr& phi) {
    if (I.rows() != I.cols() || E.rows() != E.cols()) {
        throw std::invalid_argument("RhoKsi: A and B must be square matrices");
    }
    if (phi.cols() != 1) {
        throw std::invalid_argument("RhoKsi: phi must be a column vector (N x 1)");
    }

    std::size_t m = I.rows();
    std::size_t p = E.rows();
    std::size_t N = phi.rows();

    if (m * p != N) {
        throw std::invalid_argument("RhoKsi: dim(A) * dim(B) must equal dim(phi)");
    }

    MatrixPtr psi = Psi(a, a);
    MatrixPtr tensorIE = kroneckerProduct(I, E);
    MatrixPtr outerPsi = outerProduct(psi);
    MatrixPtr rho = multiply(tensorIE, outerPsi);

    return rho;
}

MatrixPtr difference(const MatrixPtr& a, const MatrixPtr& b) {
    MatrixPtr result(a.rows(), a.cols(), MatrixPtr::Complex{0.0, 0.0});

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {

            result(i, j) = a(i, j) - b(i, j);
        }
    }
    return result;
}

MatrixPtr Conjugate(const MatrixPtr& a) {
    MatrixPtr result(a.cols(), a.rows(), MatrixPtr::Complex{0.0, 0.0});

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            std::complex<double> z = a(i, j);
            result(j, i) = std::complex<double>(z.real(), -z.imag());
        }
    }
    return result;
}

MatrixPtr createPauliX() {
    MatrixPtr X(2, 2, {0.0, 0.0});
    X(0, 1) = {1.0, 0.0};
    X(1, 0) = {1.0, 0.0};
    return X;
}

MatrixPtr createPauliY() {
    MatrixPtr Y(2, 2, {0.0, 0.0});
    Y(0, 1) = {0.0, -1.0};
    Y(1, 0) = {0.0, 1.0};
    return Y;
}

MatrixPtr createPauliZ() {
    MatrixPtr Z(2, 2, {0.0, 0.0});
    Z(0, 0) = {1.0, 0.0};
    Z(1, 1) = {-1.0, 0.0};
    return Z;
}


// 1. Деполяризующий канал
MatrixPtr depolarizingChannel(const MatrixPtr& E_ideal, double p) {
    if (p < 0 || p > 1) {
        throw std::invalid_argument("p must be between 0 and 1");
    }

    std::size_t dim = E_ideal.rows();
    if (dim != 2) {
        throw std::invalid_argument("Depolarizing channel currently only supports 2x2 matrices");
    }

    MatrixPtr result(dim, dim, {0.0, 0.0});

    // Создаем матрицы Паули
    MatrixPtr X = createPauliX();
    MatrixPtr Y = createPauliY();
    MatrixPtr Z = createPauliZ();

    // E_noisy = (1-p) * E_ideal + p/3 * (X E_ideal X + Y E_ideal Y + Z E_ideal Z)
    double coeff_ideal = 1.0 - p;
    double coeff_noise = p / 3.0;

    // Идеальная часть
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            result(i, j) = coeff_ideal * E_ideal(i, j);
        }
    }

    // Добавляем шумовые компоненты
    MatrixPtr temp = multiply(multiply(X, E_ideal), X);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            result(i, j) += coeff_noise * temp(i, j);
        }
    }

    temp = multiply(multiply(Y, E_ideal), Y);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            result(i, j) += coeff_noise * temp(i, j);
        }
    }

    temp = multiply(multiply(Z, E_ideal), Z);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            result(i, j) += coeff_noise * temp(i, j);
        }
    }

    return result;
}

// 2. Амплитудное затухание
MatrixPtr amplitudeDamping(const MatrixPtr& E_ideal, double gamma) {
    if (gamma < 0 || gamma > 1) {
        throw std::invalid_argument("gamma must be between 0 and 1");
    }

    std::size_t dim = E_ideal.rows();
    if (dim != 2) {
        throw std::invalid_argument("Amplitude damping currently only supports 2x2 matrices");
    }

    // Операторы Крауса для амплитудного затухания
    MatrixPtr E0(2, 2, {0.0, 0.0});
    MatrixPtr E1(2, 2, {0.0, 0.0});

    E0(0, 0) = {1.0, 0.0};
    E0(1, 1) = {std::sqrt(1.0 - gamma), 0.0};

    E1(0, 1) = {std::sqrt(gamma), 0.0};

    // Применяем к исходному оператору: E(ρ) = E0 ρ E0† + E1 ρ E1†
    MatrixPtr temp1 = multiply(multiply(E0, E_ideal), Conjugate(E0));
    MatrixPtr temp2 = multiply(multiply(E1, E_ideal), Conjugate(E1));

    MatrixPtr result(dim, dim, {0.0, 0.0});
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            result(i, j) = temp1(i, j) + temp2(i, j);
        }
    }

    return result;
}

// 3. Фазовый шум (дефазировка)
MatrixPtr phaseDamping(const MatrixPtr& E_ideal, double p) {
    if (p < 0 || p > 1) {
        throw std::invalid_argument("gamma must be between 0 and 1");
    }

    std::size_t dim = E_ideal.rows();
    if (dim != 2) {
        throw std::invalid_argument("Amplitude damping currently only supports 2x2 matrices");
    }

    // Операторы Крауса для фазового затухания
    MatrixPtr E0(2, 2, {0.0, 0.0});
    MatrixPtr E1(2, 2, {0.0, 0.0});

    E0(0, 0) = {1.0, 0.0};
    E0(1, 1) = {std::sqrt(1.0 - p), 0.0};

    E1(1, 1) = {std::sqrt(p), 0.0};

    // Применяем к исходному оператору: E(ρ) = E0 ρ E0† + E1 ρ E1†
    MatrixPtr temp1 = multiply(multiply(E0, E_ideal), Conjugate(E0));
    MatrixPtr temp2 = multiply(multiply(E1, E_ideal), Conjugate(E1));

    MatrixPtr result(dim, dim, {0.0, 0.0});
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            result(i, j) = temp1(i, j) + temp2(i, j);
        }
    }

    return result;
}

// 4. Случайный шум
MatrixPtr randomNoise(const MatrixPtr& E_ideal, double scale) {
    std::size_t dim = E_ideal.rows();
    MatrixPtr result(dim, dim, {0.0, 0.0});

    // Используем генератор случайных чисел
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            double real_noise = scale * dis(gen);
            double imag_noise = scale * dis(gen);

            result(i, j) = E_ideal(i, j) + MatrixPtr::Complex{real_noise, imag_noise};
        }
    }

    return result;
}

// 5. Когерентный шум (ошибка вращения вокруг оси Y)
MatrixPtr coherentNoise(const MatrixPtr& E_ideal, double theta) {
    std::size_t dim = E_ideal.rows();
    if (dim != 2) {
        throw std::invalid_argument("Coherent noise currently only supports 2x2 matrices");
    }

    // Матрица поворота вокруг оси Y
    MatrixPtr Ry(2, 2, {0.0, 0.0});
    Ry(0, 0) = {std::cos(theta/2), 0.0};
    Ry(0, 1) = {-std::sin(theta/2), 0.0};
    Ry(1, 0) = {std::sin(theta/2), 0.0};
    Ry(1, 1) = {std::cos(theta/2), 0.0};

    // Применяем поворот: Ry † E_ideal Ry
    MatrixPtr result = multiply(multiply(Ry, E_ideal), Conjugate(Ry));

    return result;
}

// 6. Универсальная функция создания зашумленного оператора
MatrixPtr createNoisyOperator(const MatrixPtr& E_ideal, double noise_level, NoiseType type) {
    switch(type) {
        case DEPOLARIZING:
            return depolarizingChannel(E_ideal, noise_level);
        case AMPLITUDE_DAMPING:
            return amplitudeDamping(E_ideal, noise_level);
        case PHASE_DAMPING:
            return phaseDamping(E_ideal, noise_level);
        case RANDOM_NOISE:
            return randomNoise(E_ideal, noise_level);
        case COHERENT_NOISE:
            return coherentNoise(E_ideal, noise_level * M_PI); // noise_level в долях π
        default:
            throw std::invalid_argument("Unknown noise type");
    }
}

// 7. Функция для получения матрицы Чоя (изоморфизм Чоя-Ямилковского)
MatrixPtr choiMatrix(const MatrixPtr& channel) {
    std::size_t d = channel.rows();
    if (d != channel.cols()) {
        throw std::invalid_argument("Channel must be square matrix");
    }

    MatrixPtr choi(d * d, d * d, {0.0, 0.0});

    // Создаем максимально запутанное состояние |Φ+> = 1/√d Σ|ii>
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
            // (I ⊗ channel)(|Φ+><Φ+|)
            for (std::size_t k = 0; k < d; ++k) {
                for (std::size_t l = 0; l < d; ++l) {
                    // Элемент матрицы Чоя: <i,k| (I ⊗ channel)(|Φ+><Φ+|) |j,l>
                    // = <i| (channel(|k><l|)) |j> / d
                    if (i == j) {
                        choi(i*d + k, j*d + l) = channel(k, l) / static_cast<double>(d);
                    }
                }
            }
        }
    }

    return choi;
}

// 8. Добавление шума через матрицу Чоя
MatrixPtr addNoiseViaChoi(const MatrixPtr& E_ideal, double noise_level) {
    std::size_t d = E_ideal.rows();

    // Получаем матрицу Чоя для идеального канала
    MatrixPtr choi_ideal = choiMatrix(E_ideal);

    // Матрица Чоя для полностью деполяризующего канала (максимальный шум)
    MatrixPtr choi_noise(d * d, d * d, {0.0, 0.0});
    for (std::size_t i = 0; i < d * d; ++i) {
        choi_noise(i, i) = {1.0 / (d * d), 0.0}; // Максимально смешанное состояние
    }

    // Смешиваем: (1-λ) * choi_ideal + λ * choi_noise
    MatrixPtr choi_mixed(d * d, d * d, {0.0, 0.0});
    double lambda = noise_level;

    for (std::size_t i = 0; i < d * d; ++i) {
        for (std::size_t j = 0; j < d * d; ++j) {
            choi_mixed(i, j) = (1.0 - lambda) * choi_ideal(i, j) + lambda * choi_noise(i, j);
        }
    }

    return choi_mixed; // Возвращаем матрицу Чоя
}

const double D(const MatrixPtr& I, const MatrixPtr& E_ideal, double noise_level,
               NoiseType noise_type, const MatrixPtr& a,
               const MatrixPtr& phi) {

    if (I.rows() != I.cols() || E_ideal.rows() != E_ideal.cols()) {
        throw std::invalid_argument("D: I and E_ideal must be square matrices");
    }
    if (phi.cols() != 1) {
        throw std::invalid_argument("D: phi must be a column vector (N x 1)");
    }

    std::size_t m = a.rows();
    std::size_t N = phi.rows();

    // Создаем зашумленный оператор
    MatrixPtr E_noisy = createNoisyOperator(E_ideal, noise_level, noise_type);

    // Вычисляем все необходимые матрицы плотности
    MatrixPtr rho11 = RhoKsi(I, E_ideal, a, phi);
    MatrixPtr rho12 = RhoKsi(I, E_noisy, a, phi);

    MatrixPtr Difference1 = difference(rho11, rho12);

    const double d1 = spectralNorm(Difference1);

    return d1;
}

// 10. Функция для исследования зависимости от уровня шума
void studyNoiseDependence(const MatrixPtr& I, const MatrixPtr& E_ideal,
                          const MatrixPtr& a, const MatrixPtr& b,
                          const MatrixPtr& phi, NoiseType noise_type) {

    std::cout << "\n=== Study of noise dependence ===" << std::endl;
    std::cout << "Noise level\tDecoherence" << std::endl;

    for (double noise = 0.0; noise <= 1.01; noise += 0.1) {
        double decoherence = D(I, E_ideal, noise, noise_type, a, phi);
        std::cout << std::fixed << std::setprecision(2) << noise << "\t\t"
                  << std::setprecision(6) << decoherence << std::endl;
    }
}

// Вспомогательная функция умножения матрицы на скаляр
MatrixPtr scalarMultiply(const MatrixPtr& mat, double factor) {
    MatrixPtr result(mat.rows(), mat.cols(), {0.0, 0.0});
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            result(i, j) = mat(i, j) * factor;
        }
    }
    return result;
}

// Вспомогательная функция для получения операторов Крауса
std::vector<MatrixPtr> getKrausOperators(NoiseType type, double level) {
    std::vector<MatrixPtr> kraus;

    switch(type) {
        case DEPOLARIZING: {
            // Деполяризующий шум: 4 оператора Паули
            MatrixPtr I(2, 2, {0.0, 0.0});
            I(0, 0) = {1.0, 0.0};
            I(1, 1) = {1.0, 0.0};

            MatrixPtr X(2, 2, {0.0, 0.0});
            X(0, 1) = {1.0, 0.0};
            X(1, 0) = {1.0, 0.0};

            MatrixPtr Y(2, 2, {0.0, 0.0});
            Y(0, 1) = {0.0, -1.0};
            Y(1, 0) = {0.0, 1.0};

            MatrixPtr Z(2, 2, {0.0, 0.0});
            Z(0, 0) = {1.0, 0.0};
            Z(1, 1) = {-1.0, 0.0};

            double p = level;
            double sqrt_p1 = std::sqrt(1.0 - p);
            double sqrt_p3 = std::sqrt(p / 3.0);

            kraus.push_back(scalarMultiply(I, sqrt_p1));
            kraus.push_back(scalarMultiply(X, sqrt_p3));
            kraus.push_back(scalarMultiply(Y, sqrt_p3));
            kraus.push_back(scalarMultiply(Z, sqrt_p3));
            break;
        }

        case PHASE_DAMPING: {
            // Фазовый шум: 2 оператора Крауса
            MatrixPtr E0(2, 2, {0.0, 0.0});
            MatrixPtr E1(2, 2, {0.0, 0.0});

            double gamma = level;
            E0(0, 0) = {1.0, 0.0};
            E0(1, 1) = {std::sqrt(1.0 - gamma), 0.0};
            E1(1, 1) = {std::sqrt(gamma), 0.0};

            kraus.push_back(E0);
            kraus.push_back(E1);
            break;
        }

        case AMPLITUDE_DAMPING: {
            // Амплитудное затухание: 2 оператора Крауса
            MatrixPtr E0(2, 2, {0.0, 0.0});
            MatrixPtr E1(2, 2, {0.0, 0.0});

            double gamma = level;
            E0(0, 0) = {1.0, 0.0};
            E0(1, 1) = {std::sqrt(1.0 - gamma), 0.0};
            E1(0, 1) = {std::sqrt(gamma), 0.0};

            kraus.push_back(E0);
            kraus.push_back(E1);
            break;
        }

        default:
            throw std::invalid_argument("Unknown noise type");
    }

    return kraus;
}


// Функция для вычисления декогеренции для заданного состояния
// D(|ψ⟩) = ||ℰ(|ψ⟩⟨ψ|) - |ψ⟩⟨ψ|||_sp
double computeDecoherence(const MatrixPtr& psi,
                          const MatrixPtr& I,
                          const MatrixPtr& E_ideal,
                          NoiseType noise_type1,
                          NoiseType noise_type2,
                          double noise_level1,
                          double noise_level2,
                          bool verbose = false) {

    if (psi.rows() != 4 || psi.cols() != 1) {
        throw std::invalid_argument("computeDecoherence: psi must be 4x1 vector");
    }

    // Шаг 1: Создаем матрицу плотности идеального состояния ρ = |ψ⟩⟨ψ|
    MatrixPtr rho_ideal = outerProduct(psi);

    // Шаг 2: Получаем операторы Крауса для шумов
    std::vector<MatrixPtr> kraus1 = getKrausOperators(noise_type1, noise_level1);
    std::vector<MatrixPtr> kraus2 = getKrausOperators(noise_type2, noise_level2);

    // Шаг 3: Применяем шумы: ℰ(ρ) = Σ_{i,j} (E_i ⊗ F_j) ρ (E_i ⊗ F_j)†
    MatrixPtr rho_noisy(4, 4, {0.0, 0.0});

    for (const auto& E : kraus1) {
        for (const auto& F : kraus2) {
            // Составной оператор Крауса: K = E ⊗ F
            MatrixPtr K = kroneckerProduct(E, F);

            // K ρ K†
            MatrixPtr Krho = multiply(K, rho_ideal);
            MatrixPtr KrhoKdag = multiply(Krho, Conjugate(K));

            // Суммируем
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    rho_noisy(i, j) += KrhoKdag(i, j);
                }
            }
        }
    }

    // Шаг 4: Вычисляем разность Δ = ℰ(ρ) - ρ
    MatrixPtr delta = difference(rho_noisy, rho_ideal);

    // Шаг 5: Вычисляем спектральную норму ||Δ||_sp
    double decoherence = spectralNorm(delta);

    if (verbose) {
        std::cout << "Decoherence computed: " << decoherence << std::endl;
    }

    return decoherence;
}


void normalizeState(MatrixPtr& psi) {
    if (psi.rows() != 4 || psi.cols() != 1) {
        throw std::invalid_argument("normalizeState: psi must be 4x1 vector");
    }

    double norm = 0.0;
    for (int i = 0; i < 4; ++i) {
        norm += std::norm(psi(i, 0));
    }
    norm = std::sqrt(norm);

    if (norm > 1e-12) {
        for (int i = 0; i < 4; ++i) {
            psi(i, 0) /= norm;
        }
    }
}

// Функция для вывода разложения состояния
void printStateDecomposition(const MatrixPtr& psi) {
    if (psi.rows() != 4 || psi.cols() != 1) {
        std::cout << "Invalid state" << std::endl;
        return;
    }

    std::cout << "psi = ";
    bool first = true;
    std::string basis[4] = {"00", "01", "10", "11"};

    for (int i = 0; i < 4; ++i) {
        std::complex<double> coeff = psi(i, 0);
        if (std::abs(coeff) > 1e-12) {
            if (!first) {
                if (coeff.real() >= 0 && coeff.imag() >= 0) {
                    std::cout << " + ";
                } else if (coeff.real() < 0 && coeff.imag() < 0) {
                    std::cout << " - ";
                } else if (coeff.real() >= 0 && coeff.imag() < 0) {
                    std::cout << " + ";
                } else {
                    std::cout << " - ";
                }
            } else if (coeff.real() < 0 || coeff.imag() < 0) {
                std::cout << "-";
            }

            double real_abs = std::abs(coeff.real());
            double imag_abs = std::abs(coeff.imag());

            if (std::abs(real_abs - 1.0) < 1e-12 && std::abs(imag_abs) < 1e-12) {
                // Коэффициент = 1 или -1, ничего не выводим
            } else if (std::abs(imag_abs) < 1e-12) {
                std::cout << real_abs;
            } else if (std::abs(real_abs) < 1e-12) {
                std::cout << imag_abs << "i";
            } else {
                std::cout << "(" << real_abs << (coeff.imag() >= 0 ? "+" : "-") << imag_abs << "i)";
            }

            std::cout << basis[i];
            first = false;
        }
    }

    if (first) {
        std::cout << "0";
    }
    std::cout << std::endl;
}

// Генерация случайного состояния
MatrixPtr randomState() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    MatrixPtr psi(4, 1, {0.0, 0.0});
    for (int i = 0; i < 4; ++i) {
        psi(i, 0) = std::complex<double>(dis(gen), dis(gen));
    }
    normalizeState(psi);

    return psi;
}



// Вычисление градиента численным методом
// Возвращает градиент как вектор той же размерности (4x1)
MatrixPtr computeGradient(const MatrixPtr& psi,
                          const MatrixPtr& I,
                          const MatrixPtr& E_ideal,
                          NoiseType noise_type1,
                          NoiseType noise_type2,
                          double noise_level1,
                          double noise_level2,
                          double epsilon = 1e-6) {

    if (psi.rows() != 4 || psi.cols() != 1) {
        throw std::invalid_argument("computeGradient: psi must be 4x1 vector");
    }

    MatrixPtr grad(4, 1, {0.0, 0.0});

    // Текущее значение функции
    double current_value = computeDecoherence(psi, I, E_ideal,
                                               noise_type1, noise_type2,
                                               noise_level1, noise_level2);

    // Для каждой компоненты вектора (4 компоненты)
    for (int i = 0; i < 4; ++i) {
        // Градиент по действительной части
        MatrixPtr psi_plus_real = psi;
        psi_plus_real(i, 0) += std::complex<double>(epsilon, 0.0);
        normalizeState(psi_plus_real);
        double value_real_plus = computeDecoherence(psi_plus_real, I, E_ideal,
                                                     noise_type1, noise_type2,
                                                     noise_level1, noise_level2);

        // Градиент по мнимой части
        MatrixPtr psi_plus_imag = psi;
        psi_plus_imag(i, 0) += std::complex<double>(0.0, epsilon);
        normalizeState(psi_plus_imag);
        double value_imag_plus = computeDecoherence(psi_plus_imag, I, E_ideal,
                                                     noise_type1, noise_type2,
                                                     noise_level1, noise_level2);

        // Записываем градиент как комплексное число
        grad(i, 0) = std::complex<double>((value_real_plus - current_value) / epsilon,
                                          (value_imag_plus - current_value) / epsilon);
    }

    return grad;
}


// Функция для вычисления спектральной нормы матрицы
double spectralNormMatrix(const MatrixPtr& matrix) {
    // Для матрицы 4x4 используем упрощенный метод
    if (matrix.rows() != 4 || matrix.cols() != 4) {
        throw std::invalid_argument("spectralNormMatrix: matrix must be 4x4");
    }

    // Вычисляем сингулярные числа через собственные значения A†A
    MatrixPtr A(4, 4, {0.0, 0.0});
    MatrixPtr A_dag = Conjugate(matrix);

    // A = matrix† * matrix
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                A(i, j) += A_dag(i, k) * matrix(k, j);
            }
        }
    }

    // Для 4x4 матрицы используем степенной метод
    return spectralNorm(A);
}







// Тихая версия computeDecoherence без вывода
double computeDecoherenceSilent(const MatrixPtr& psi,
                                 const MatrixPtr& I,
                                 const MatrixPtr& E_ideal,
                                 NoiseType noise_type1,
                                 NoiseType noise_type2,
                                 double noise_level1,
                                 double noise_level2) {

    // Перенаправляем вывод в никуда для подавления лишних сообщений
    std::streambuf* old_cout = std::cout.rdbuf();
    std::stringstream null_stream;
    std::cout.rdbuf(null_stream.rdbuf());

    double result = computeDecoherence(psi, I, E_ideal,
                                        noise_type1, noise_type2,
                                        noise_level1, noise_level2,
                                        false);

    // Восстанавливаем вывод
    std::cout.rdbuf(old_cout);

    return result;
}

// Функция для вычисления нормы градиента (для отладки)
double computeGradientNorm(const MatrixPtr& psi,
                           const MatrixPtr& I,
                           const MatrixPtr& E_ideal,
                           NoiseType noise_type1,
                           NoiseType noise_type2,
                           double noise_level1,
                           double noise_level2,
                           double epsilon = 1e-6) {

    MatrixPtr grad = computeGradient(psi, I, E_ideal,
                                      noise_type1, noise_type2,
                                      noise_level1, noise_level2,
                                      epsilon);

    double norm = 0.0;
    for (int i = 0; i < 4; ++i) {
        norm += std::norm(grad(i, 0));
    }
    return std::sqrt(norm);
}

// Функция вычисления нового приближения
MatrixPtr calculateNewState(const MatrixPtr& psi,
                            const MatrixPtr& grad,
                            double lambda,
                            bool maximize = true) {

    MatrixPtr psi_new(4, 1, {0.0, 0.0});

    if (maximize) {
        // Для максимизации: psi_new = psi + lambda * grad
        for (int i = 0; i < 4; ++i) {
            psi_new(i, 0) = psi(i, 0) + lambda * grad(i, 0);
        }
    } else {
        // Для минимизации: psi_new = psi - lambda * grad
        for (int i = 0; i < 4; ++i) {
            psi_new(i, 0) = psi(i, 0) - lambda * grad(i, 0);
        }
    }

    normalizeState(psi_new);

    return psi_new;
}

// Метод наискорейшего подъема (исправленная версия)
MatrixPtr gradAscent(const MatrixPtr& start_psi,
                     double eps,
                     const MatrixPtr& I,
                     const MatrixPtr& E_ideal,
                     NoiseType noise_type1,
                     NoiseType noise_type2,
                     double noise_level1,
                     double noise_level2,
                     double& best_value,
                     int max_iterations = 1000,
                     bool verbose = true) {

    MatrixPtr current = start_psi;
    MatrixPtr last(4, 1, {0.0, 0.0});
    double last_value;
    double current_value;
    double learning_rate = 0.1;  // Начальный шаг

    current_value = computeDecoherenceSilent(current, I, E_ideal,
                                              noise_type1, noise_type2,
                                              noise_level1, noise_level2);

    if (verbose) {
        std::cout << "Initial decoherence: " << current_value << std::endl;
        std::cout << "Initial state: ";
        printStateDecomposition(current);
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Сохраняем текущее состояние
        for (int i = 0; i < 4; ++i) {
            last(i, 0) = current(i, 0);
        }
        last_value = current_value;

        // Вычисляем градиент
        MatrixPtr grad = computeGradient(current, I, E_ideal,
                                         noise_type1, noise_type2,
                                         noise_level1, noise_level2);

        // Проверяем норму градиента
        double grad_norm = 0.0;
        for (int i = 0; i < 4; ++i) {
            grad_norm += std::norm(grad(i, 0));
        }
        grad_norm = std::sqrt(grad_norm);

        if (grad_norm < eps) {
            if (verbose) {
                std::cout << "Gradient norm too small: " << grad_norm << std::endl;
            }
            break;
        }

        // Нормируем градиент
        for (int i = 0; i < 4; ++i) {
            grad(i, 0) /= grad_norm;
        }

        // Линейный поиск оптимального шага
        double lambda = learning_rate;
        double best_lambda = 0.0;
        double best_dec = current_value;

        // Пробуем разные шаги
        for (int step = 0; step < 10; ++step) {
            MatrixPtr test_state = calculateNewState(current, grad, lambda, true);
            double test_dec = computeDecoherenceSilent(test_state, I, E_ideal,
                                                        noise_type1, noise_type2,
                                                        noise_level1, noise_level2);
            if (test_dec > best_dec) {
                best_dec = test_dec;
                best_lambda = lambda;
            }
            lambda *= 0.5;  // Уменьшаем шаг
        }

        if (best_lambda == 0.0) {
            if (verbose) {
                std::cout << "No improvement found" << std::endl;
            }
            break;
        }

        // Обновляем состояние
        current = calculateNewState(current, grad, best_lambda, true);
        current_value = best_dec;

        if (verbose && (iter + 1) % 10 == 0) {
            std::cout << "Iteration " << iter + 1 << ": decoherence = " << current_value
                      << ", lambda = " << best_lambda << std::endl;
        }

        // Проверка сходимости
        if (std::abs(current_value - last_value) < eps) {
            if (verbose) {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            }
            break;
        }

        // Адаптивно изменяем learning rate
        if (best_lambda > 0.05) {
            learning_rate = std::min(0.5, learning_rate * 1.1);
        } else if (best_lambda < 0.01) {
            learning_rate = std::max(0.001, learning_rate * 0.9);
        }
    }

    best_value = current_value;

    if (verbose) {
        std::cout << "Final decoherence: " << current_value << std::endl;
        std::cout << "Final state: ";
        printStateDecomposition(current);
    }

    return current;
}

// Функция для поиска состояния с максимальной декогеренцией
MatrixPtr findWorstCaseState(const MatrixPtr& I,
                              const MatrixPtr& E_ideal,
                              NoiseType noise_type1,
                              NoiseType noise_type2,
                              double noise_level1,
                              double noise_level2,
                              double& max_decoherence,
                              double eps = 1e-8,
                              int num_restarts = 10,
                              int max_iterations = 500,
                              bool verbose = true) {

    MatrixPtr best_state(4, 1, {0.0, 0.0});
    max_decoherence = -1.0;

    std::cout << "\n=== Searching for worst-case state ===" << std::endl;
    std::cout << "Noise type 1: " << noise_type1 << ", level: " << noise_level1 << std::endl;
    std::cout << "Noise type 2: " << noise_type2 << ", level: " << noise_level2 << std::endl;
    std::cout << "Restarts: " << num_restarts << std::endl;
    std::cout << "=====================================\n" << std::endl;

    for (int restart = 0; restart < num_restarts; ++restart) {
        if (verbose) {
            std::cout << "--- Restart " << restart + 1 << " ---" << std::endl;
        }

        // Генерируем случайное начальное состояние
        MatrixPtr start = randomState();

        // Запускаем градиентный подъем
        double decoherence_value;
        MatrixPtr result = gradAscent(start, eps, I, E_ideal,
                                      noise_type1, noise_type2,
                                      noise_level1, noise_level2,
                                      decoherence_value,
                                      max_iterations,
                                      verbose);

        if (verbose) {
            std::cout << "Restart " << restart + 1 << " finished. Decoherence = " << decoherence_value << std::endl;
        }

        if (decoherence_value > max_decoherence) {
            max_decoherence = decoherence_value;
            best_state = result;
            if (verbose) {
                std::cout << "*** New best found! ***" << std::endl;
            }
        }
        if (verbose) {
            std::cout << std::endl;
        }
    }

    return best_state;
}

MatrixPtr reducedDensityFirst(const MatrixPtr& psi) {
    // Проверяем, что psi - вектор 4x1
    if (psi.rows() != 4 || psi.cols() != 1) {
        throw std::invalid_argument("reducedDensityFirst: psi must be 4x1");
    }

    // Результат - матрица 2x2 для первого кубита
    MatrixPtr rho(2, 2, {0.0, 0.0});

    // Частичный след по второму кубиту
    // ρ_A(i,i') = Σ_{j=0,1} ψ(i,j) · ψ*(i',j)
    for (int i = 0; i < 2; ++i) {        // i - индекс первого кубита в ρ_A
        for (int i_prime = 0; i_prime < 2; ++i_prime) {
            for (int j = 0; j < 2; ++j) {  // j - индекс второго кубита (по которому берем след)
                // Индексы в 4-мерном векторе: i*2 + j
                rho(i, i_prime) += psi(i * 2 + j, 0) * std::conj(psi(i_prime * 2 + j, 0));
            }
        }
    }

    return rho;
}

MatrixPtr reducedDensitySecond(const MatrixPtr& psi) {
    if (psi.rows() != 4 || psi.cols() != 1) {
        throw std::invalid_argument("reducedDensitySecond: psi must be 4x1");
    }

    MatrixPtr rho(2, 2, {0.0, 0.0});

    // Частичный след по первому кубиту
    // ρ_B(j,j') = Σ_{i=0,1} ψ(i,j) · ψ*(i,j')
    for (int j = 0; j < 2; ++j) {
        for (int j_prime = 0; j_prime < 2; ++j_prime) {
            for (int i = 0; i < 2; ++i) {
                rho(j, j_prime) += psi(i * 2 + j, 0) * std::conj(psi(i * 2 + j_prime, 0));
            }
        }
    }

    return rho;
}

struct BlochVector {
    double x, y, z;
};

BlochVector computeBlochVector(const MatrixPtr& rho) {
    BlochVector bloch;
    // ρ = (I + x·σ_x + y·σ_y + z·σ_z) / 2
    bloch.x = 2.0 * rho(0, 1).real();
    bloch.y = 2.0 * rho(0, 1).imag();
    bloch.z = rho(0, 0).real() - rho(1, 1).real();
    return bloch;
}
// Функция для проверки свойства: Tr₂[ℰ(|ψ⟩⟨ψ|)] = ℰ₁(ρ₁)
// где ρ₁ = Tr₂(|ψ⟩⟨ψ|) — редуцированное состояние первого кубита
bool verifyPartialTracePropertyWithPsi(const MatrixPtr& psi,      // 4x1 вектор состояния
                                        const MatrixPtr& I,       // единичная матрица 2x2
                                        NoiseType noise_type1,    // тип шума на первом кубите
                                        NoiseType noise_type2,    // тип шума на втором кубите
                                        double noise_level1,      // уровень шума на первом кубите
                                        double noise_level2,      // уровень шума на втором кубите
                                        double tolerance = 1e-10) {

    // Проверка размерности
    if (psi.rows() != 4 || psi.cols() != 1) {
        throw std::invalid_argument("psi must be 4x1 vector");
    }

    // ===== Шаг 1: Создаем матрицу плотности полного состояния ρ = |ψ⟩⟨ψ| =====
    MatrixPtr rho_total = outerProduct(psi);  // 4x4 матрица

    // ===== Шаг 2: Применяем полный составной шум ℰ = ℰ₁ ⊗ ℰ₂ =====
    std::vector<MatrixPtr> kraus1 = getKrausOperators(noise_type1, noise_level1);
    std::vector<MatrixPtr> kraus2 = getKrausOperators(noise_type2, noise_level2);

    MatrixPtr rho_noisy_total(4, 4, {0.0, 0.0});

    for (const auto& E : kraus1) {
        for (const auto& F : kraus2) {
            MatrixPtr K = kroneckerProduct(E, F);  // K = E ⊗ F
            MatrixPtr Krho = multiply(K, rho_total);
            MatrixPtr KrhoKdag = multiply(Krho, Conjugate(K));

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    rho_noisy_total(i, j) += KrhoKdag(i, j);
                }
            }
        }
    }

    // ===== Шаг 3: Берем частичный след по второму кубиту =====
    // Tr₂(ρ_noisy_total) — результат 2x2 матрица
    MatrixPtr rho_A_partial_trace(2, 2, {0.0, 0.0});

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                // Индексы: (i*2 + k, j*2 + k)
                rho_A_partial_trace(i, j) += rho_noisy_total(i * 2 + k, j * 2 + k);
            }
        }
    }

    // ===== Шаг 4: Вычисляем редуцированное состояние первого кубита из исходного =====
    // ρ₁ = Tr₂(|ψ⟩⟨ψ|)
    MatrixPtr rho1(2, 2, {0.0, 0.0});

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                rho1(i, j) += psi(i * 2 + k, 0) * std::conj(psi(j * 2 + k, 0));
            }
        }
    }

    // ===== Шаг 5: Применяем шум только к первому кубиту ℰ₁(ρ₁) =====
    std::vector<MatrixPtr> kraus1_only = getKrausOperators(noise_type1, noise_level1);

    MatrixPtr rho1_noisy(2, 2, {0.0, 0.0});

    for (const auto& E : kraus1_only) {
        MatrixPtr Erho = multiply(E, rho1);
        MatrixPtr ErhoEdag = multiply(Erho, Conjugate(E));

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                rho1_noisy(i, j) += ErhoEdag(i, j);
            }
        }
    }

    // ===== Шаг 6: Сравниваем результаты =====
    // Вычисляем разность: Tr₂[ℰ(|ψ⟩⟨ψ|)] - ℰ₁(ρ₁)
    MatrixPtr diff = difference(rho_A_partial_trace, rho1_noisy);

    // Вычисляем норму разности
    double norm_diff = spectralNorm(diff);

    // Выводим результаты
    std::cout << "\n check" << std::endl;
    std::cout << "initial state  ";
    printStateDecomposition(psi);
    std::cout << std::endl;
    std::cout << "reduced rho1" << std::endl;
    std::cout << "[" << rho1(0, 0) << ", " << rho1(0, 1) << "]" << std::endl;
    std::cout << "[" << rho1(1, 0) << ", " << rho1(1, 1) << "]" << std::endl;
    std::cout << std::endl;
    std::cout << "noise1 " << noise_type1 << ", level " << noise_level1 << std::endl;
    std::cout << "noise2  " << noise_type2 << ", level " << noise_level2 << std::endl;
    std::cout << " norm diffrence " << norm_diff << std::endl;

    if (norm_diff < tolerance) {
        std::cout << "rule works" << std::endl;
        std::cout << "  discrepancy " << tolerance << ")" << std::endl;
        return true;
    } else {
        std::cout << "doesnt work" << std::endl;
        std::cout << "  discrepancy " << norm_diff << " > " << tolerance << std::endl;
        return false;
    }
}

// Функция для проверки свойства на найденном наихудшем состоянии
void testPartialTraceOnWorstState(const MatrixPtr& worst_state,
                                    const MatrixPtr& I,
                                    NoiseType noise_type1,
                                    NoiseType noise_type2,
                                    double noise_level1,
                                    double noise_level2) {

    std::cout << "\n========================================" << std::endl;
    std::cout << "******** CHECK THE TRACE RULE on our wors state ********** " << std::endl;
    std::cout << "========================================" << std::endl;

    bool passed = verifyPartialTracePropertyWithPsi(worst_state, I,
                                                     noise_type1, noise_type2,
                                                     noise_level1, noise_level2);

    // Дополнительная проверка: также вычислим вектор Блоха для проверки
    MatrixPtr rho1(2, 2, {0.0, 0.0});
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                rho1(i, j) += worst_state(i * 2 + k, 0) * std::conj(worst_state(j * 2 + k, 0));
            }
        }
    }

}

// Функция для проверки с несколькими случайными состояниями
void testPartialTraceWithRandomStates(const MatrixPtr& I,
                                       NoiseType noise_type1,
                                       NoiseType noise_type2,
                                       double noise_level1,
                                       double noise_level2,
                                       int num_tests = 5,
                                       bool verbose = false) {

    std::cout << "\n    check of many random states   " << std::endl;
    std::cout << " number of tests " << num_tests << std::endl;

    bool all_passed = true;

    for (int test = 0; test < num_tests; ++test) {
        // Генерируем случайное 4-мерное состояние
        MatrixPtr psi = randomState();

        if (verbose) {
            std::cout << "\n test " << test + 1 << " ---" << std::endl;
            std::cout << " state  ";
            printStateDecomposition(psi);
        }

        bool passed = verifyPartialTracePropertyWithPsi(psi, I,
                                                         noise_type1, noise_type2,
                                                         noise_level1, noise_level2,
                                                         1e-8);

        if (!passed) {
            all_passed = false;
            if (!verbose) {
                std::cout << "\n  test " << test + 1 << " failed " << std::endl;
                std::cout << " state ";
                printStateDecomposition(psi);
            }
        }
    }

    std::cout << "\n  conclusion" << std::endl;
    if (all_passed) {
        std::cout << "all " << num_tests << " good " << std::endl;
        std::cout << " works for all" << std::endl;
    } else {
        std::cout << "no" << std::endl;
    }
}
void saveBlochDataForPython_A(const MatrixPtr& worst_state, const std::string& filename = "bloch_data_A.txt") {
    //MatrixPtr rho = outerProduct(psi);
    //BlochVector bloch = computeBlochVector(rho);

    MatrixPtr rho_A = reducedDensityFirst(worst_state);

    BlochVector bloch_A = computeBlochVector(rho_A);

    std::ofstream file(filename);
    if (file.is_open()) {
        file << "# Bloch vector data for visualization\n";
        file << "# x y z\n";
        file << bloch_A.x << " " << bloch_A.y << " " << bloch_A.z << "\n";
        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    }
}

void saveBlochDataForPython_B(const MatrixPtr& worst_state, const std::string& filename = "bloch_data_B.txt") {
    //MatrixPtr rho = outerProduct(psi);
    //BlochVector bloch = computeBlochVector(rho);

    MatrixPtr rho_B = reducedDensitySecond(worst_state);

    BlochVector bloch_B = computeBlochVector(rho_B);

    std::ofstream file(filename);
    if (file.is_open()) {
        file << "# Bloch vector data for visualization\n";
        file << "# x y z\n";
        file << bloch_B.x << " " << bloch_B.y << " " << bloch_B.z << "\n";
        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    }
}

// ============================================================================
// НОВЫЕ ФУНКЦИИ ДЛЯ ПОШАГОВОГО МОДЕЛИРОВАНИЯ УРАВНЕНИЯ ЛИНДБЛАДА
// ============================================================================

// ============================================================================
// ИСПРАВЛЕННЫЙ МЕТОД ОПЕРАТОРОВ КРАУСА (СТАБИЛЬНАЯ ВЕРСИЯ)
// ============================================================================
// ============================================================================
// ИСПРАВЛЕННЫЙ МЕТОД ОПЕРАТОРОВ КРАУСА (РАБОТАЮЩАЯ ВЕРСИЯ)
// ============================================================================

struct LindbladParams {
    double epsilon_B;      // туннельное расщепление кубита B (Дж)
    double Gamma_B;        // скорость релаксации на кубите B (1/с)
    double Gamma_phi;      // скорость дефазировки на кубите A (1/с)
    double hbar;           // приведённая постоянная Планка

    LindbladParams() : hbar(1.054571817e-34) {}
};

// Гамильтониан CNOT (4x4) в Дж
MatrixPtr createCNOTHamiltonianSI(double epsilon_B) {
    MatrixPtr H(4, 4, MatrixPtr::Complex{0.0, 0.0});
    double coeff = -epsilon_B / 2.0;
    H(2, 3) = MatrixPtr::Complex(coeff, 0.0);
    H(3, 2) = MatrixPtr::Complex(coeff, 0.0);
    return H;
}

// Операторы
MatrixPtr createSigmaZA() {
    MatrixPtr sz(4, 4, MatrixPtr::Complex{0.0, 0.0});
    sz(0, 0) = MatrixPtr::Complex(1.0, 0.0);
    sz(1, 1) = MatrixPtr::Complex(1.0, 0.0);
    sz(2, 2) = MatrixPtr::Complex(-1.0, 0.0);
    sz(3, 3) = MatrixPtr::Complex(-1.0, 0.0);
    return sz;
}

MatrixPtr createSigmaMinusB() {
    MatrixPtr sm(4, 4, MatrixPtr::Complex{0.0, 0.0});
    sm(0, 2) = MatrixPtr::Complex(1.0, 0.0);
    sm(1, 3) = MatrixPtr::Complex(1.0, 0.0);
    return sm;
}

MatrixPtr createSigmaPlusB() {
    MatrixPtr sp(4, 4, MatrixPtr::Complex{0.0, 0.0});
    sp(2, 0) = MatrixPtr::Complex(1.0, 0.0);
    sp(3, 1) = MatrixPtr::Complex(1.0, 0.0);
    return sp;
}

MatrixPtr createIdentity4() {
    MatrixPtr I4(4, 4, MatrixPtr::Complex{0.0, 0.0});
    for (int i = 0; i < 4; ++i) I4(i, i) = MatrixPtr::Complex(1.0, 0.0);
    return I4;
}

// ============================================================================
// ОПЕРАТОРЫ КРАУСА
// ============================================================================

struct KrausOperators {
    MatrixPtr K0;
    MatrixPtr K1;
    MatrixPtr K2;

    KrausOperators() : K0(4, 4), K1(4, 4), K2(4, 4) {}
};

KrausOperators getKrausOperatorsSmallStep(const LindbladParams& params, double dt) {
    KrausOperators kraus;

    // dt в секундах
    double dtau = dt / params.hbar;  // безразмерное время

    // Гамильтониан в размерности частоты (рад/с)
    MatrixPtr H_J = createCNOTHamiltonianSI(params.epsilon_B);
    MatrixPtr H(4, 4, MatrixPtr::Complex{0.0, 0.0});
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double h_real = H_J(i, j).real() / params.hbar;
            double h_imag = H_J(i, j).imag() / params.hbar;
            H(i, j) = MatrixPtr::Complex(h_real, h_imag);
        }
    }

    // Операторы Линдблада
    MatrixPtr smB = createSigmaMinusB();
    MatrixPtr spB = createSigmaPlusB();
    MatrixPtr szA = createSigmaZA();
    MatrixPtr I4 = createIdentity4();

    // L₁†L₁ = σ_+σ_-
    MatrixPtr L1dL1 = multiply(spB, smB);

    // Сумма L†L
    MatrixPtr sumLdL(4, 4, MatrixPtr::Complex{0.0, 0.0});
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double val_real = L1dL1(i, j).real() * params.Gamma_B
                            + I4(i, j).real() * params.Gamma_phi / 2.0;
            double val_imag = L1dL1(i, j).imag() * params.Gamma_B
                            + I4(i, j).imag() * params.Gamma_phi / 2.0;
            sumLdL(i, j) = MatrixPtr::Complex(val_real, val_imag);
        }
    }

    // H_eff = H - i/2 * sumLdL
    MatrixPtr Heff(4, 4, MatrixPtr::Complex{0.0, 0.0});
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double real_part = H(i, j).real() + 0.5 * sumLdL(i, j).imag();
            double imag_part = H(i, j).imag() - 0.5 * sumLdL(i, j).real();
            Heff(i, j) = MatrixPtr::Complex(real_part, imag_part);
        }
    }

    // K0 = I - i * H_eff * dtau
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double real_part = (i == j ? 1.0 : 0.0) + Heff(i, j).imag() * dtau;
            double imag_part = -Heff(i, j).real() * dtau;
            kraus.K0(i, j) = MatrixPtr::Complex(real_part, imag_part);
        }
    }

    // K1 = √(Γ_B * dt) * σ_-
    double sqrt_gammaB_dt = std::sqrt(params.Gamma_B * dt);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double real_part = smB(i, j).real() * sqrt_gammaB_dt;
            double imag_part = smB(i, j).imag() * sqrt_gammaB_dt;
            kraus.K1(i, j) = MatrixPtr::Complex(real_part, imag_part);
        }
    }

    // K2 = √(Γ_φ * dt) * σ_z / √2
    double sqrt_gammaPhi_dt = std::sqrt(params.Gamma_phi * dt);
    double norm = 1.0 / std::sqrt(2.0);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double real_part = szA(i, j).real() * sqrt_gammaPhi_dt * norm;
            double imag_part = szA(i, j).imag() * sqrt_gammaPhi_dt * norm;
            kraus.K2(i, j) = MatrixPtr::Complex(real_part, imag_part);
        }
    }

    return kraus;
}

// Применение шага
MatrixPtr applyKrausStep(const MatrixPtr& rho, const KrausOperators& kraus) {
    MatrixPtr result(4, 4, MatrixPtr::Complex{0.0, 0.0});

    MatrixPtr K0_rho = multiply(kraus.K0, rho);
    MatrixPtr term0 = multiply(K0_rho, Conjugate(kraus.K0));

    MatrixPtr K1_rho = multiply(kraus.K1, rho);
    MatrixPtr term1 = multiply(K1_rho, Conjugate(kraus.K1));

    MatrixPtr K2_rho = multiply(kraus.K2, rho);
    MatrixPtr term2 = multiply(K2_rho, Conjugate(kraus.K2));

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::complex<double> val = term0(i, j) + term1(i, j) + term2(i, j);
            result(i, j) = MatrixPtr::Complex(val.real(), val.imag());
        }
    }

    return result;
}

// ============================================================================
// МОДЕЛИРОВАНИЕ
// ============================================================================

struct TimeEvolutionResult {
    std::vector<double> times;
    std::vector<double> prob_B1;
    std::vector<double> decoherence;
    double optimal_time;
    double max_prob_B1;
};

TimeEvolutionResult simulateCNOTimeEvolution(const MatrixPtr& psi_initial,
                                              const LindbladParams& params,
                                              double t_max,
                                              double dt) {

    TimeEvolutionResult result;

    MatrixPtr rho = outerProduct(psi_initial);
    MatrixPtr rho_ideal = rho;

    double omega = params.epsilon_B / (2.0 * params.hbar);
    double t_ideal = M_PI / omega;

    std::cout << "\n=== Simulating CNOT time evolution (Kraus) ===" << std::endl;
    std::cout << "t_max = " << t_max * 1e12 << " ps" << std::endl;
    std::cout << "dt = " << dt * 1e15 << " fs" << std::endl;
    std::cout << "epsilon_B = " << params.epsilon_B / 1.602e-19 << " eV" << std::endl;
    std::cout << "Gamma_B = " << params.Gamma_B << " s^-1" << std::endl;
    std::cout << "Gamma_phi = " << params.Gamma_phi << " s^-1" << std::endl;
    std::cout << "Ideal time = " << t_ideal * 1e12 << " ps" << std::endl;
    std::cout << "===============================================" << std::endl;

    double t = 0.0;
    double max_P1 = 0.0;
    double opt_t = 0.0;
    int step = 0;
    int print_every = 100;

    while (t <= t_max && step < 100000) {
        double P1 = rho(3, 3).real();

        if (step % print_every == 0) {
            std::cout << "t = " << t * 1e12 << " ps, P1 = " << P1 << std::endl;
        }

        result.times.push_back(t);
        result.prob_B1.push_back(P1);

        MatrixPtr delta = difference(rho, rho_ideal);
        result.decoherence.push_back(spectralNorm(delta));

        if (P1 > max_P1 && !std::isnan(P1) && P1 <= 1.0) {
            max_P1 = P1;
            opt_t = t;
        }

        if (std::isnan(P1)) {
            std::cout << "NaN at step " << step << std::endl;
            break;
        }

        KrausOperators kraus = getKrausOperatorsSmallStep(params, dt);
        rho = applyKrausStep(rho, kraus);

        t += dt;
        step++;
    }

    result.optimal_time = opt_t;
    result.max_prob_B1 = max_P1;

    std::cout << "\n=== Simulation complete ===" << std::endl;
    std::cout << "Max P(|1⟩_B) = " << max_P1 << " at t = " << opt_t * 1e12 << " ps" << std::endl;

    return result;
}

void demonstrateTimeEvolution() {
    MatrixPtr psi_initial(4, 1, MatrixPtr::Complex{0.0, 0.0});
    psi_initial(2, 0) = MatrixPtr::Complex(1.0, 0.0);

    LindbladParams params;
    params.epsilon_B = 1.0e-22;
    params.Gamma_B = 1.0e8;
    params.Gamma_phi = 1.0e7;

    double omega = params.epsilon_B / (2.0 * params.hbar);
    double t_ideal = M_PI / omega;
    double t_max = 2.0 * t_ideal;
    double dt = t_ideal / 1000.0;  // 1000 шагов

    TimeEvolutionResult result = simulateCNOTimeEvolution(psi_initial, params, t_max, dt);

    std::ofstream file("time_evolution.txt");
    if (file.is_open()) {
        for (size_t i = 0; i < result.times.size(); ++i) {
            file << result.times[i] * 1e12 << " " << result.prob_B1[i] << "\n";
        }
        file.close();
    }
}


int main() {

        // Инициализация базовых состояний
        MatrixPtr a(2, 1);
        a(0, 0) = {1.0, 0.0};

        MatrixPtr b(2, 1);
        b(1, 0) = {1.0, 0.0};

        MatrixPtr I(2, 2);
        I(0, 0) = {1.0, 0.0};
        I(1, 1) = {1.0, 0.0};

        MatrixPtr E_ideal(2, 2);
        E_ideal(0, 1) = {0.0, -1.0};
        E_ideal(1, 0) = {0.0, 1.0};

        // Параметры шума
        double noise_level1 = 0.3;
        double noise_level2 = 0.3;
        NoiseType noise_type1 = AMPLITUDE_DAMPING;
        NoiseType noise_type2 = AMPLITUDE_DAMPING;

        // Поиск наихудшего состояния (максимальная декогеренция)
        double max_decoherence;
        MatrixPtr worst_state = findWorstCaseState(I, E_ideal,
                                                    noise_type1, noise_type2,
                                                    noise_level1, noise_level2,
                                                    max_decoherence,
                                                    1e-8, 10, 500);

        std::cout << "\n=== WORST-CASE STATE FOUND ===" << std::endl;
        std::cout << "Maximum decoherence: " << max_decoherence << std::endl;
        std::cout << "State: ";
        printStateDecomposition(worst_state);

    MatrixPtr rho_A = reducedDensityFirst(worst_state);
    MatrixPtr rho_B = reducedDensitySecond(worst_state);

    BlochVector bloch_A = computeBlochVector(rho_A);
    BlochVector bloch_B = computeBlochVector(rho_B);

    std::cout << "Qubit A Bloch vector: (" << bloch_A.x << ", " << bloch_A.y << ", " << bloch_A.z << ")" << std::endl;
    std::cout << "Qubit B Bloch vector: (" << bloch_B.x << ", " << bloch_B.y << ", " << bloch_B.z << ")" << std::endl;


    testPartialTraceOnWorstState(worst_state, I,
                                     noise_type1, noise_type2,
                                     noise_level1, noise_level2);

    // Проверка на нескольких случайных состояниях
    testPartialTraceWithRandomStates(I,
                                      noise_type1, noise_type2,
                                      noise_level1, noise_level2,
                                      5, false);


    // 2. Сохранение данных для Python визуализации
    saveBlochDataForPython_A(worst_state, "bloch_data_A.txt");
    saveBlochDataForPython_B(worst_state, "bloch_data_B.txt");

   demonstrateTimeEvolution();
        return 0;
    }


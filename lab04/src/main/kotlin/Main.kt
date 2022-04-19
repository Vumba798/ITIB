import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.Multik.math
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.math.exp
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.abs
import kotlin.math.exp

val x_4 = mk.ndarray(mk[
        mk[0, 0, 0, 0],
        mk[0, 0, 0, 1],
        mk[0, 0, 1, 0],
        mk[0, 0, 1, 1],
        mk[0, 1, 0, 0],
        mk[0, 1, 0, 1],
        mk[0, 1, 1, 0],
        mk[0, 1, 1, 1],
        mk[1, 0, 0, 0],
        mk[1, 0, 0, 1],
        mk[1, 0, 1, 0],
        mk[1, 0, 1, 1],
        mk[1, 1, 0, 0],
        mk[1, 1, 0, 1],
        mk[1, 1, 1, 0],
        mk[1, 1, 1, 1]])

val t_4 = mk.ndarray(mk[
0, 0, 0, 0,
1, 1, 0, 1,
0, 0, 0, 0,
0, 0, 0, 0])

fun f1(net: Double) = if (net >= 0.0) 1 else 0
fun f2(net: Double) = if (0.5 * (net / (1 + abs(net)) + 1) >= 0.5) 1 else 0


class RbfNetwork(trainIndexes: List<Int>, private val f: (Double) -> Int) {
    private val x = x_4.toListD2().withIndex()
        .filter {
            it.index in trainIndexes
        }
        .map {
            it.value
        }
        .toNDArray()

    private val t = t_4.data.withIndex()
        .filter {
            it.index in trainIndexes
        }
        .map {
            it.value
        }
        .toNDArray()
    private val J0: Int = t_4.data.count { it == 0 }
    private val J1: Int = t_4.data.count { it == 1 }
    private val indexes = t_4.data.withIndex()
        .filter {
            it.value == if (J1 < J0) 1 else 0
        }
        .map {
            it.index
        }
    private val J: Int = minOf(J0, J1)
    private val eta = 0.3
    private val weights = mk.d1array(J + 1) { .0 }
    private val phi = mk.d1array(J) { .0 }
    private val centers: D2Array<Int> by lazy {
        val sublist = x_4.toListD2().withIndex()
            .filter {
                it.index in indexes
            }
            .map {
                it.value
            }
        mk.ndarray(sublist).asD2Array()
    }

    fun test() {
        val errorVector = mk.d1array(t_4.size) { 0 }
        for (i in 0 until x_4.shape[0]) {
            val output = calculate(x_4[i])
            val target = t_4[i]
            errorVector[i] = target - output
        }
        println(errorVector)
        return
    }

    fun train() {
        println("weights: $weights")
        var epochError = -1
        var deltaVector = mk.d1array(J) { 0 }
        var epoch = 0
        while (epochError != 0) {
            epochError = 0
            for (i in 0 until x.shape[0]) {
                val output = calculate(x[i])
                val delta = t[i] - output
                epochError += abs(delta)
                fit(delta)
            }
            println("Epoch: $epoch, epoch error: $epochError")
            epoch++
        }
        println("Weights: $weights")
    }
    private fun calculate(input: MultiArray<Int, D1>): Int{
        // TODO refatctor with activation function argument
        for (j in 0 until J) {
            val squaredDiff = (centers[j] - input).map { it * it }
            phi[j] = exp(-squaredDiff.sum().toDouble())
        }
        val net = (weights[1..weights.size] * phi + weights[0]).sum()
        return f(net)
    }
    private fun fit(delta: Int) {
        for (i in 0 until J) {
            weights[i + 1] += eta * delta * phi[i]
        }
        weights[0] += eta * delta * 1
    }
}

fun main(args: Array<String>) {
    val trainIndexes1 = listOf(0, 4, 6, 7)
    val obj1 = RbfNetwork(trainIndexes1, ::f1)
    obj1.train()
    obj1.test()

    val trainIndexes2 = listOf(4, 6, 7)
    val obj2 = RbfNetwork(trainIndexes2, ::f2)
    obj2.train()
    obj2.test()
}

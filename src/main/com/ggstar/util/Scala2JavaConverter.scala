package com.ggstar.util

import ml.combust.mleap.tensor.Tensor

object Scala2JavaConverter {
  def pauseCtr(prob:Tensor[Double]):java.lang.Double = {
    println("ctr", prob.get(1).head)
    prob.get(1).head
  }
}

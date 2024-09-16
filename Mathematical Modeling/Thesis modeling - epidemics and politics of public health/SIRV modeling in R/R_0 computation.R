# parameters
gamma <- 0.2
beta <- 0.1
cm <- matrix(c(5.8,1.3,2.6,1.9),ncol=2,byrow = TRUE)
dist <- c(1/3,2/3)
# calculate next generation matrix
V <- diag(rep(gamma,2))
F1 <- cm
for (i in 1:2) {
  for (j in 1:2) {
    F1[i,j] <- (dist[i]/dist[j])*beta*F1[i,j]
  }
}
FF <- F1
# K is the next generation matrix
K <- FF %*% solve(V)
ee <- eigen(K)
R0 <- max(abs(Re(ee$values)))
cat("R0:",R0)



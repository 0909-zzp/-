# 全量图形 Lasso（并行加速版）
library(glasso)
library(foreach)
library(doParallel)

# 设置路径
data_dir <- "C:/Users/27438/Desktop/大创/1min"

# 参数
lambda <- 0.0001
eps <- 1e-6
K <- 392
maxit <- 2000      # 适当减少迭代次数
thr <- 1e-5        # 放宽收敛容差

# 获取文件列表
files <- list.files(data_dir, pattern = "_1min_log_return.RData$", full.names = TRUE)
files <- sort(files)
n_days <- length(files)
cat("共", n_days, "个交易日\n")

# 并行设置（根据 CPU 核心数调整，留一个给系统）
n_cores <- parallel::detectCores() - 1
cat("使用", n_cores, "个核心\n")
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# 并行处理
results <- foreach(day = 1:n_days, .packages = "glasso") %dopar% {
  load(files[day])
  returns <- t(rett1)                 # (390, K)
  cov_raw <- crossprod(returns)
  diag(cov_raw) <- diag(cov_raw) + eps
  gl <- glasso(cov_raw, rho = lambda, maxit = maxit, thr = thr)
  reg_cov <- gl$w
  ones <- rep(1, K)
  w <- solve(reg_cov) %*% ones
  w <- w / sum(w)
  cond_val <- kappa(reg_cov, exact = TRUE)
  list(w = w, cond = cond_val)
}

stopCluster(cl)

# 提取结果
reg_weights <- do.call(rbind, lapply(results, `[[`, "w"))
cond_vals <- sapply(results, `[[`, "cond")

# 单独计算最后一天的精度矩阵（用于网络）
cat("计算最后一天的精度矩阵...\n")
last_file <- files[n_days]
load(last_file)
returns_last <- t(rett1)
cov_last <- crossprod(returns_last)
diag(cov_last) <- diag(cov_last) + eps
gl_last <- glasso(cov_last, rho = lambda, maxit = maxit, thr = thr)
prec_last <- gl_last$wi

# 保存结果
write.csv(reg_weights, "C:/Users/27438/Desktop/图形Lasso/reg_weights_100.csv", row.names = FALSE)
write.csv(data.frame(cond = cond_vals), "C:/Users/27438/Desktop/图形Lasso/reg_cond_100.csv", row.names = FALSE)
write.csv(prec_last, "C:/Users/27438/Desktop/图形Lasso/prec_last_100.csv", row.names = FALSE)

cat("全量结果已保存至","\n")


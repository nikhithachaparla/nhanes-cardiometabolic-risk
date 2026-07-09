# nhanes_survey_analysis.R
# Survey-weighted estimation for the NHANES 2021-2023 cardiometabolic paper.

library(survey)
library(dplyr)

df <- read.csv("nhanes_analytic_clean.csv")

options(survey.lonely.psu = "adjust")
des <- svydesign(ids = ~psu, strata = ~strata,
                 weights = ~survey_weight, nest = TRUE, data = df)

# Weighted prevalence with 95% CIs
for (v in c("undiagnosed_diabetes", "undiagnosed_prediabetes",
            "undiagnosed_hypertension", "undiagnosed_hyperlipidemia")) {
  est <- svyciprop(as.formula(paste0("~", v)), des, method = "logit")
  cat(sprintf("%-30s %4.1f%%  (95%% CI %4.1f%%--%4.1f%%)\n",
              v, coef(est)*100, confint(est)[1]*100, confint(est)[2]*100))
}

# Subgroup CIs
svyby(~undiagnosed_prediabetes, ~race_ethnicity, des, svyciprop, vartype="ci", method="logit")
svyby(~er_visit_binary, ~risk_tier, des, svyciprop, vartype="ci", method="logit")

# Insurance subgroup
des_ins <- subset(des, insurance_label %in% c("Insured","Uninsured"))
svyby(~undiagnosed_hypertension, ~insurance_label, des_ins, svyciprop, vartype="ci", method="logit")

# Design-based tests
m1 <- svyglm(undiagnosed_prediabetes ~ race_ethnicity, design=des, family=quasibinomial())
regTermTest(m1, ~race_ethnicity)

m2 <- svyglm(undiagnosed_hypertension ~ insurance_label, design=des_ins, family=quasibinomial())
regTermTest(m2, ~insurance_label)

# Adjusted model - modified Poisson for prevalence ratios
mod <- svyglm(er_visit_binary ~ risk_tier + age + gender + race_ethnicity +
                poverty_income_ratio + education + has_insurance +
                usual_care_site + bmi,
              design=des, family=quasipoisson(link="log"))

se <- sqrt(diag(vcov(mod)))
pr_table <- data.frame(
  PR      = round(exp(coef(mod)), 3),
  CI_low  = round(exp(coef(mod) - 1.96*se), 3),
  CI_high = round(exp(coef(mod) + 1.96*se), 3)
)
print(pr_table)

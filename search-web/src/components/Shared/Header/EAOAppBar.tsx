import { AppBar, Divider, Grid, Typography } from "@mui/material";
import EAO_Logo from "@/assets/images/EAO_Logo.png";
import { AppConfig } from "@/utils/config";
import AppBarActions from "./AppBarActions";
import { useIsMobile } from "@/hooks/common";
import { BCDesignTokens } from "epic.theme";

export default function EAOAppBar() {
  const isMobile = useIsMobile();
  return (
    <>
      <AppBar
        position="fixed"
        color="inherit"
        sx={{
          borderBottom: `1px solid ${BCDesignTokens.surfaceColorBorderDefault}`,
          boxShadow: "none",
        }}
      >
        <Grid
          container
          marginY={BCDesignTokens.layoutMarginSmall}
          paddingX={isMobile ? 0 : "0.5rem"}
          justifyContent="space-between"
        >
          <Grid item display="flex" justifyContent="start" alignItems="center">
            <img src={EAO_Logo} height={isMobile ? 40 : 56} />
            {!isMobile && (
              <>
                <Divider orientation="vertical" flexItem sx={{ m: 1 }} />
                <Typography
                  variant="h2"
                  color="inherit"
                  component="div"
                  paddingLeft={"0.5rem"}
                  fontWeight={"bold"}
                >
                  {AppConfig.appTitle || "EPIC.centre"}
                </Typography>
              </>
            )}
          </Grid>
          <Grid
            item
            display="flex"
            justifyContent="center"
            alignItems="center"
            paddingRight={"0.75rem"}
          >
            <AppBarActions />
          </Grid>
        </Grid>
      </AppBar>
    </>
  );
}

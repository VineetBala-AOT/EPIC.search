import { useState, useMemo } from "react";
import {
  Box,
  Divider,
  IconButton,
  InputBase,
  Typography,
  Collapse,
  List,
  ListItem,
  Paper,
  Button,
} from "@mui/material";
import CloseIcon from '@mui/icons-material/Close';
import {
  Search as SearchIcon,
  ExpandLess,
  ExpandMore,
  ChevronLeft,
  ChevronRight,
} from "@mui/icons-material";
import { CheckboxGroup, Checkbox } from "@bcgov/design-system-react-components";

interface FilterSidebarProps {
  rawProjects: any[];
  rawDocTypes: any;
  selectedProjects: string[];
  selectedDocTypes: string[];
  onProjectToggle: (projectId: string) => void;
  onDocTypeToggle: (docTypeId: string) => void;
  onClearAll: () => void;
  collapsed?: boolean;
  onCollapseToggle?: () => void;
}

export function FilterSidebar({
  rawProjects,
  rawDocTypes,
  selectedProjects,
  selectedDocTypes,
  onProjectToggle,
  onDocTypeToggle,
  onClearAll,
  collapsed = false,
  onCollapseToggle,
}: FilterSidebarProps) {

  const [projectSearch, setProjectSearch] = useState("");
  const [docTypeSearch, setDocTypeSearch] = useState("");
  const [expandedActs, setExpandedActs] = useState<string[]>([]);
  const isCollapsed = collapsed;

  const projectList = useMemo(() => {
    return rawProjects.map((p: any) => ({
      id: p.project_id,
      name: p.project_name,
    }));
  }, [rawProjects]);

  const docTypes = useMemo(() => {
    const root = (rawDocTypes as any)?.result?.document_types;

    if (!root) return [];

    return Object.entries(root).map(([id, value]: any) => ({
      id,
      name: value.name,
      act: value.act.includes("2018") ? "2018" : "2002",
      aliases: value.aliases ?? [],
    }));
  }, [rawDocTypes]);

  // Filtered projects
  const filteredProjects = projectList.filter((p) =>
    p.name.toLowerCase().includes(projectSearch.toLowerCase())
  );

  // Filtered document types
  const docTypes2018 = docTypes.filter((d) => d.act === "2018");
  const docTypes2002 = docTypes.filter((d) => d.act === "2002");
  const filteredDocTypes2018 = docTypes2018.filter((d) =>
    d.name.toLowerCase().includes(docTypeSearch.toLowerCase())
  );
  const filteredDocTypes2002 = docTypes2002.filter((d) =>
    d.name.toLowerCase().includes(docTypeSearch.toLowerCase())
  );

  const toggleAct = (act: string) => {
    setExpandedActs((prev) =>
      prev.includes(act) ? prev.filter((a) => a !== act) : [...prev, act]
    );
  };

  const activeFilters = [
    ...selectedProjects.map((id) => ({
      id,
      label: projectList.find((p) => p.id === id)?.name || "",
      type: "project" as const,
    })),
    ...selectedDocTypes.map((id) => ({
      id,
      label: docTypes.find((d) => d.id === id)?.name || "",
      type: "docType" as const,
    })),
  ];

  return (
    <>
      {/* Overlay for mobile */}
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          paddingTop: "20px",
          width: "100%",
        }}
      >
        {!isCollapsed && (
          <Typography variant="subtitle2" sx={{ ml: 2, fontWeight: 600 }}>
            PROJECTS
          </Typography>
        )}
        <IconButton
            onClick={() => onCollapseToggle?.()}
            size="small"
            sx={isCollapsed ? { ml: 1 } : undefined}
            >
            {isCollapsed ? <ChevronRight /> : <ChevronLeft />}
        </IconButton>
      </Box>
      {!isCollapsed && (
        <Box sx={{ p: 2 }}>
          <Paper
            elevation={0}
            sx={{
              p: "2px 8px",
              mb: -1,
              display: "flex",
              alignItems: "center",
              backgroundColor: "#fff",
              border: "1px solid #d1d5db",
              borderRadius: "0.25rem",
              transition: "border 0.2s",
              "&:hover": {
                border: "1px solid #013366",
              },
            }}
          >
            <SearchIcon fontSize="small" sx={{ mr: 1 }} />
            <InputBase
              placeholder="Search projects..."
              value={projectSearch}
              onChange={(e) => setProjectSearch(e.target.value)}
              sx={{
                flex: 1,
                typography: "body2",
              }}
            />
          </Paper>

          <CheckboxGroup
            value={selectedProjects}
            onChange={(values) => {
              values.forEach((id) => {
                if (!selectedProjects.includes(id)) {
                  onProjectToggle(id);
                }
              });

              selectedProjects.forEach((id) => {
                if (!values.includes(id)) {
                  onProjectToggle(id);
                }
              });
            }}
          >
            <List
              sx={{
                width: "100%",
                maxHeight: 48 * 6,
                overflowY: "auto",
                pl: 0,
                pr: 0,
                mt: 1,
              }}
            >
              {filteredProjects.map((d) => (
                <ListItem
                  key={d.id}
                  sx={{
                    alignItems: "center",
                    py: 1, // spacing between rows
                    pl: 0.4,
                  }}
                >
                  <Checkbox value={d.id}>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1.25, // space between checkbox and text
                      }}
                    >
                      <Typography variant="body2">{d.name}</Typography>
                    </Box>
                  </Checkbox>
                </ListItem>
              ))}
            </List>
          </CheckboxGroup>

          {/* Document Types */}
          <Typography variant="subtitle2" sx={{ mt: 2, mb: 1, fontWeight: 600 }}>
            DOCUMENT TYPES
          </Typography>
          <Paper
            elevation={0}
            sx={{
              p: "2px 8px",
              mb: 1.5,
              display: "flex",
              alignItems: "center",
              backgroundColor: "#fff",
              border: "1px solid #d1d5db",
              borderRadius: "0.25rem",
              transition: "border 0.2s",
              "&:hover": {
                border: "1px solid #013366",
              },
            }}
          >
            <SearchIcon fontSize="small" sx={{ mr: 1 }} />
            <InputBase
              placeholder="Search document types..."
              value={docTypeSearch}
              onChange={(e) => setDocTypeSearch(e.target.value)}
              sx={{ flex: 1, typography: "body2" }}
            />
          </Paper>

          {/* 2002 Act */}
          <Paper
            elevation={0}
            sx={{
              mb: 1.5,
              backgroundColor: "#fff",
              border: "1px solid #d1d5db",
              borderRadius: "0.25rem",
            }}
          >
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                p: 1,
                cursor: "pointer",
              }}
              onClick={() => toggleAct("2002")}
            >
              <Typography variant="body2">2002 Act</Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  ({filteredDocTypes2002.length})
                </Typography>
                {expandedActs.includes("2002") ? <ExpandLess /> : <ExpandMore />}
              </Box>
            </Box>
            <Divider/>
            <Collapse in={expandedActs.includes("2002")}>
              <CheckboxGroup
                value={selectedDocTypes}
                onChange={(values) => {
                  values.forEach((id) => {
                    if (!selectedDocTypes.includes(id)) {
                      onDocTypeToggle(id);
                    }
                  });

                  selectedDocTypes.forEach((id) => {
                    if (!values.includes(id)) {
                      onDocTypeToggle(id);
                    }
                  });
                }}
              >
                <List sx={{ maxHeight: 48 * 6, overflowY: "auto"}}>
                  {filteredDocTypes2002.map((d) => (
                    <ListItem
                      key={d.id}
                      sx={{
                        alignItems: "center",
                        py: 1, // spacing between rows
                      }}
                    >
                      <Checkbox value={d.id}>
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            gap: 1.25, // space between checkbox and text
                          }}
                        >
                          <Typography variant="body2">{d.name}</Typography>
                        </Box>
                      </Checkbox>
                    </ListItem>
                  ))}
                </List>
              </CheckboxGroup>
            </Collapse>
          </Paper>

          {/* 2018 Act */}
          <Paper
            elevation={0}
            sx={{
              mb: 1,
              backgroundColor: "#fff",
              border: "1px solid #d1d5db",
              borderRadius: "0.25rem",
            }}
          >
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                p: 1,
                cursor: "pointer",
              }}
              onClick={() => toggleAct("2018")}
            >
              <Typography variant="body2">2018 Act</Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  ({filteredDocTypes2018.length})
                </Typography>
                {expandedActs.includes("2018") ? <ExpandLess /> : <ExpandMore />}
              </Box>
            </Box>
            <Divider/>
            <Collapse in={expandedActs.includes("2018")}>
              <CheckboxGroup
                value={selectedDocTypes}
                onChange={(values) => {
                  values.forEach((id) => {
                    if (!selectedDocTypes.includes(id)) {
                      onDocTypeToggle(id);
                    }
                  });

                  selectedDocTypes.forEach((id) => {
                    if (!values.includes(id)) {
                      onDocTypeToggle(id);
                    }
                  });
                }}
              >
                <List sx={{ maxHeight: 48 * 6, overflowY: "auto"}}>
                  {filteredDocTypes2018.map((d) => (
                    <ListItem
                      key={d.id}
                      sx={{
                        alignItems: "center",
                        py: 1, // spacing between rows
                      }}
                    >
                      <Checkbox value={d.id}>
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            gap: 1.25, // space between checkbox and text
                          }}
                        >
                          <Typography variant="body2">{d.name}</Typography>
                        </Box>
                      </Checkbox>
                    </ListItem>
                  ))}
                </List>
              </CheckboxGroup>
            </Collapse>
          </Paper>

          {/* Active Filters */}
          {activeFilters.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                ACTIVE FILTERS
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 1 }}>
                {activeFilters.map((f) => (
                  <Paper
                    key={`${f.type}-${f.id}`}
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      px: 1.5,
                      py: 0.5,
                      bgcolor: "#D6E9FB",
                      borderRadius: "16px",
                      gap: 0.5,
                    }}
                  >
                    <Typography variant="caption">{f.label}</Typography>
                    <IconButton
                      size="small"
                      onClick={() =>
                        f.type === "project"
                          ? onProjectToggle(f.id)
                          : onDocTypeToggle(f.id)
                      }
                    >
                      <CloseIcon sx={{ fontSize: 14 }} />
                    </IconButton>
                  </Paper>
                ))}
              </Box>
              <Box sx={{ ml: -1.5 }}>
                <Button
                  variant="text"
                  size="small"
                  onClick={onClearAll}
                  sx={{ textDecoration: "underline", color: "#013366" }}
                >
                  Clear All Filters
                </Button>
              </Box>
            </Box>
          )}
        </Box>
      )}
    </>
  );
}

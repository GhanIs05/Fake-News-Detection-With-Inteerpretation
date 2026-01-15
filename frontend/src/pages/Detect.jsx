// import {useState} from "react";
// import DetectionLayout from "../components/DetectionLayout";
// import TextDetector from "../components/TextDetector.jsx";
// import ImageDetector from "../components/ImageDetector.jsx";
// import ResultCard from "../components/ResultCard.jsx";

// export default function Detect() {
//     const [activeTab, setActiveTab] = useState("text");
//     const [result, setResult] = useState(null);
  
//     return (
//     <DetectionLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      
//       {activeTab === "text" && (
//         <TextDetector onResult={setResult} />
//       )}
      

//       {activeTab === "image" && (
//         <ImageDetector onResult={setResult} />
//       )}

//       {/* Result appears below detector */}
//       {result && <ResultCard result={result} />}

//     </DetectionLayout>
//   );
// }
import { useState } from "react";
import DetectionLayout from "../components/DetectionLayout";
import TextDetector from "../components/TextDetector.jsx";
import ImageDetector from "../components/ImageDetector.jsx";
import ResultCard from "../components/ResultCard.jsx";

export default function Detect() {
  const [activeTab, setActiveTab] = useState("text");
  const [result, setResult] = useState(null);
  const [model, setModel] = useState("bert"); // ✅ default best model

  return (
    <DetectionLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      
      {activeTab === "text" && (
        <TextDetector
          onResult={setResult}
          model={model}
          setModel={setModel}
        />
      )}

      {activeTab === "image" && (
        <ImageDetector onResult={setResult} />
      )}
      <div className="mt-10" />
      {/* Result appears below detector */}
      {result && <ResultCard result={result} />}

    </DetectionLayout>
  );
}

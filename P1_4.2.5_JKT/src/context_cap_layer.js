// DeckGL-native contextual cap layer.
//
// The cap mesh is shared and instanced, but its texture UVs are NOT derived
// from a per-cell rotation/affine transform. Each instance supplies the atlas
// UV of its four fixed geographic footprint corners. The vertex shader then
// interpolates directly across those corner values. Cap Z moves each epoch;
// the corner geography (and therefore the local map context) never moves.

import {SimpleMeshLayer} from '@deck.gl/mesh-layers';

export class ContextCapLayer extends SimpleMeshLayer {
  static layerName = 'ContextCapLayer';

  initializeState() {
    super.initializeState();
    this.getAttributeManager().addInstanced({
      // [SW.u, SW.v, SE.u, SE.v]
      instanceUvSouth: {
        type: 'float32',
        size: 4,
        accessor: 'getContextUvSouth',
        defaultValue: [0, 1, 1, 1],
      },
      // [NW.u, NW.v, NE.u, NE.v]
      instanceUvNorth: {
        type: 'float32',
        size: 4,
        accessor: 'getContextUvNorth',
        defaultValue: [0, 0, 1, 0],
      },
    });
  }

  getShaders() {
    const shaders = super.getShaders();
    return {
      ...shaders,
      inject: {
        ...(shaders.inject ?? {}),
        'vs:#decl': `
          in vec4 instanceUvSouth;
          in vec4 instanceUvNorth;
        `,
        'vs:#main-end': `
          // texCoords belong to the shared local cap quad: (0,0)=SW and
          // (1,1)=NE. Interpolate the actual geographic atlas UV at each
          // corner, so a rotated RUM receives the correct north-up map.
          vec2 localCapUv = texCoords;
          vec2 atlasUvSouth = mix(instanceUvSouth.xy, instanceUvSouth.zw, localCapUv.x);
          vec2 atlasUvNorth = mix(instanceUvNorth.xy, instanceUvNorth.zw, localCapUv.x);
          vTexCoord = mix(atlasUvSouth, atlasUvNorth, localCapUv.y);
          geometry.uv = vTexCoord;
        `,
        // Sample the B/W map, then combine the live deformation tint or the
        // blankie grey veil in the SAME opaque cap draw. This prevents the
        // far-view coplanar z-fighting that a second polygon veil created.
        'fs:#main-end': `
          if (!bool(picking.isActive)) {
            vec4 contextTexel = texture(sampler, clamp(vTexCoord, vec2(0.0), vec2(1.0)));
            float tintAlpha = clamp(vColor.a, 0.0, 1.0);
            vec4 contextComposite = vec4(
              mix(contextTexel.rgb, vColor.rgb, tintAlpha),
              1.0
            );
            fragColor = picking_filterHighlightColor(contextComposite);
          }
        `,
      },
    };
  }
}
